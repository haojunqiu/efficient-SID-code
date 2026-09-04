"""Image-level denoiser: specializes the closed-form denoiser to images.

A ``ClosedFormDenoiser`` (see ``closed_form_denoiser.py``) denoises a set of D-dimensional
vectors. ``ImageDenoiser`` makes that operate on a whole image: unfold the image into overlapping
patches, denoise the patch vectors with the closed-form denoiser, then fold the results back with
Gaussian weighting. ``extract_patches`` here is the unfold half of that pair (and is also reused to
build the dataset of patches the closed-form denoiser is fit to).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from efficient_sid.closed_form_denoiser import ClosedFormDenoiser, ClosedFormDenoiserConfig
from efficient_sid.scheduler import Scheduler
from efficient_sid.timing import GPUTime


def extract_patches(
    image: torch.Tensor,
    patch_size: int,
    stride: int = 1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract overlapping patches from a (C, H, W) image.

    Returns (N, C, patch_size, patch_size).
    If *mask* is provided (H, W), only patches fully inside valid (True) regions are kept.
    """
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(image.size(0), -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)

    if mask is not None:
        mask_patches = mask.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        mask_patches = mask_patches.contiguous().view(-1, patch_size, patch_size)
        valid_indices = mask_patches.bool().all(dim=(1, 2))
        patches = patches[valid_indices]

    return patches


@dataclass
class ImageDenoiserConfig:
    """``ImageDenoiser``'s settings, and the patch denoiser it wraps.

    ``dataset_stride``, ``query_stride`` and ``fold_rho`` are each a scalar or a list with one
    entry per pyramid scale, so none can be typed: OmegaConf cannot express
    `Union[int, List[int]]`.
    """
    patch_size: int = 15      # the square patch side; larger patches copy larger structures
    dataset_stride: Any = 1   # stride over the patches extracted into the dataset; 1 = every patch
    query_stride: Any = 1     # stride over the noisy patches denoised each step; also takes "auto"
    fold_rho: Any = 0.2       # Gaussian width the overlapping results are folded back with
    patch_denoiser: ClosedFormDenoiserConfig = field(default_factory=ClosedFormDenoiserConfig)


class ImageDenoiser(torch.nn.Module):
    """Wraps a closed-form (patch) denoiser to operate on full images.

    Pipeline: extract patches → closed-form denoiser → Gaussian-weighted fold → normalize.

    ``patch_denoiser`` arrives already built from ``config.patch_denoiser``; see
    ``build.build_image_denoiser``. The channel count is read back from its ``D`` and
    ``patch_size``, so it is never configured here.
    """

    def __init__(
        self,
        config: ImageDenoiserConfig,
        patch_denoiser: ClosedFormDenoiser,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.patch_denoiser = patch_denoiser
        # D is one flattened (C, patch_size, patch_size) patch.
        assert patch_denoiser.D % config.patch_size ** 2 == 0
        self.C = patch_denoiser.D // config.patch_size ** 2
        self.patch_size = config.patch_size
        self.stride = (config.query_stride, config.query_stride)
        self.fold_rho = config.fold_rho
        self.dtype = dtype
        # image_time is the whole image-level denoiser (unfold -> kernel -> fold); patch_time
        # is just the patch kernel inside it, so the difference is the unfold/fold overhead.
        self.image_time = GPUTime()
        self.patch_time = GPUTime()
        self.register_buffer(
            'weights',
            self._gaussian_weights(config.fold_rho).to(dtype)
        )

    def _gaussian_weights(self, rho: float = 0.2) -> torch.Tensor:
        """Build a separable Gaussian window over the patch's normalized [-1, 1] coordinates.

        Small rho concentrates weight at the patch center, so overlapping patches blend
        mostly by their centers; rho -> inf flattens the window to uniform weighting.
        """
        gh = torch.linspace(-1, 1, self.patch_size, dtype=torch.float32)
        gw = torch.linspace(-1, 1, self.patch_size, dtype=torch.float32)
        nh = torch.exp(-0.5 * (gh / rho).pow(2))
        nw = torch.exp(-0.5 * (gw / rho).pow(2))
        weights = torch.einsum('i,j->ij', nh, nw)
        weights = weights.repeat(self.C, 1, 1)
        return weights.view(1, self.C * self.patch_size ** 2)

    def _fold(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """Sum overlapping patches back into an image of ``output_size``."""
        return F.fold(
            x,
            output_size=output_size,
            kernel_size=self.patch_size,
            stride=self.stride,
        )

    @torch.inference_mode()
    def forward(self, xt: torch.Tensor, t: int, scheduler: Scheduler) -> torch.Tensor:
        """
        xt: (1, C, H, W)
        t:  int timestep index, 1..num_steps (index 0 is ``scheduler``'s terminal)
        Returns: (1, C, H, W) denoised prediction.
        """
        assert scheduler.sigmas[t] > 0, "the kernel scale 1/sigma**2 is undefined at sigma = 0"
        with self.image_time.record():
            return self._forward(xt, t, scheduler)

    def _forward(self, xt: torch.Tensor, t: int, scheduler: Scheduler) -> torch.Tensor:
        assert len(xt) == 1
        B, C, H, W = xt.shape

        xt_patches = extract_patches(
            xt[0],
            patch_size=self.patch_size,
            stride=self.stride[0],
        ).flatten(1)

        with self.patch_time.record():
            hat_xt_patches = self.patch_denoiser(
                xt=xt_patches,
                alpha=scheduler.alphas[t],
                sigma=scheduler.sigmas[t],
            )

        hat_xt_patches = hat_xt_patches * self.weights
        weights_expanded = self.weights.expand(len(hat_xt_patches), -1)

        # Fold adds over overlaps, so folding the weights the same way and dividing turns the
        # sum into a weighted average -- the image reconstruction step of the paper.
        weights_sum = self._fold(
            weights_expanded.flatten(1).permute(1, 0).unsqueeze(0),
            output_size=(H, W))
        image_out = self._fold(
            hat_xt_patches.flatten(1).permute(1, 0).unsqueeze(0),
            output_size=(H, W))
        image_out = image_out / weights_sum

        if self.stride[0] != 1 or self.stride[1] != 1:
            return torch.nan_to_num(image_out)
        return image_out.to(self.dtype)

    @torch.inference_mode()
    def forward_shift(
        self,
        xt: torch.Tensor,
        t: int,
        shift: str,
        scheduler: Scheduler,
    ) -> torch.Tensor:
        """Denoise on a circularly shifted view: roll, denoise, un-roll.

        Tileable sampling averages over these, so that every region -- including the one that
        straddles the wrap -- is denoised somewhere away from an edge.

        Parameters
        ----------
        xt : Tensor
            (1, C, H, W) noisy image at step ``t``.
        t : int
            Scheduler step index.
        shift : str
            'no_shift' — identity, denoise in place.
            'horiz_half' — roll by W//2 along width.
            'vert_half' — roll by H//2 along height.
            'diag_half' — roll by both, which covers the corners.
        scheduler : Scheduler
            Supplies ``alpha`` and ``sigma`` at ``t``.

        Returns
        -------
        Tensor
            The clean estimate, un-rolled back to the original alignment.
        """
        H, W = xt.shape[-2], xt.shape[-1]
        shifts_map = {
            'no_shift': (0, 0),
            'horiz_half': (0, W // 2),
            'vert_half': (H // 2, 0),
            'diag_half': (H // 2, W // 2),
        }
        shifts = shifts_map[shift]
        if shifts == (0, 0):
            return self.forward(xt, t, scheduler)
        xt_shifted = torch.roll(xt, shifts=shifts, dims=(2, 3))
        out_shifted = self.forward(xt_shifted, t, scheduler)
        return torch.roll(out_shifted, shifts=tuple(-s for s in shifts), dims=(2, 3))
