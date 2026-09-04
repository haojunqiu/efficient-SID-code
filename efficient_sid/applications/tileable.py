"""Tileable generation: produce seamlessly tileable textures.

Algorithm
---------
Standard coarse-to-fine sampling where each denoiser call is replaced by
``blend_shifts()``: the image is circularly shifted, denoised from
multiple viewpoints, and blended using spatial weight masks so that edges
(which become tile seams) are predicted accurately.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch

from efficient_sid.image_denoiser import ImageDenoiser
from efficient_sid.latent import FluxVAE, maybe_decode
from efficient_sid.pyramid import BasePyramidProcessor
from efficient_sid.scheduler import Scheduler
from efficient_sid.timing import timed_compute
from efficient_sid.utils import save_scale_result, unnormalize_to_zero_to_one


@dataclass
class TilingConfig:
    """Which seams to close, and how."""
    direction: str = "both"          # "horizontal", "vertical", or "both"
    num_shifts: int = 4              # 4 adds the diagonal shift; the paper used 3
    periodic_decode: bool = True     # latent only: decode with circular-padded convs, which
                                     # keep a seamless latent seamless in pixels


def seam_error(img: torch.Tensor, tiling_direction: str) -> float:
    """Measure the wrap-around discontinuity in 0-255 units: how visible the seam is when tiled.

    Compares the gradient *across* the wrap (last column vs first column, and/or last row vs first
    row) against the median gradient *inside* the image, and reports the excess. ~0 means the tile
    seam is no sharper than ordinary image detail, i.e. seamless.
    """
    def _excess(x: torch.Tensor, dim: int) -> torch.Tensor:
        wrap = (x.index_select(dim, torch.tensor([0], device=x.device))
                - x.index_select(dim, torch.tensor([x.shape[dim] - 1], device=x.device))).abs().mean()
        interior = x.diff(dim=dim).abs().median()
        return (wrap - interior).clamp(min=0).item() * 255.0 / 2.0

    errs = []
    if tiling_direction in ("horizontal", "both"):
        errs.append(_excess(img, -1))
    if tiling_direction in ("vertical", "both"):
        errs.append(_excess(img, -2))
    return max(errs)


def blend_shifts(
    denoiser: ImageDenoiser,
    scheduler: Scheduler,
    xt: torch.Tensor,
    t: int,
    tiling_direction: str,
    num_shifts: int = 4,
) -> torch.Tensor:
    """Blend denoiser outputs from multiple circular shifts.

    Each shift places different content at the image center where the denoiser
    is most accurate. The results are blended with complementary spatial masks.

    Parameters
    ----------
    denoiser : ImageDenoiser
        Must have a ``forward_shift(xt, t, shift, scheduler)`` method.
    scheduler : Scheduler
        The one for this denoiser's scale.
    xt : Tensor (1, C, H, W)
    t : int
    tiling_direction : str
        'horizontal', 'vertical', or 'both'.
    num_shifts : int
        3 = horizontal + vertical + center, the setting used for the paper's results.
        4 adds a diagonal shift, which covers the corners: more seamless tiles for one extra
        denoiser call per step, and what the shipped tileable configs use. Only affects 'both'.

    Returns
    -------
    Tensor (1, C, H, W) — blended denoised prediction.
    """
    H, W = xt.shape[-2], xt.shape[-1]

    if tiling_direction == "horizontal":
        # Shifted version is accurate at left/right edges; non-shifted at center
        w_shifted = torch.ones_like(xt)
        w_shifted[..., W // 4: 3 * W // 4] = 0.0
        weights = [w_shifted, 1.0 - w_shifted]
        shift_names = ["horiz_half", "no_shift"]

    elif tiling_direction == "vertical":
        w_shifted = torch.ones_like(xt)
        w_shifted[..., H // 4: 3 * H // 4, :] = 0.0
        weights = [w_shifted, 1.0 - w_shifted]
        shift_names = ["vert_half", "no_shift"]

    elif tiling_direction == "both":
        if num_shifts == 4:
            # 4-shift: horiz handles left/right edges, vert handles top/bottom edges,
            # diag handles corners, center handles interior
            w_horiz = torch.zeros_like(xt)
            w_horiz[..., H // 4: 3 * H // 4, :W // 4] = 1.0
            w_horiz[..., H // 4: 3 * H // 4, 3 * W // 4:] = 1.0

            w_vert = torch.zeros_like(xt)
            w_vert[..., :H // 4, W // 4: 3 * W // 4] = 1.0
            w_vert[..., 3 * H // 4:, W // 4: 3 * W // 4] = 1.0

            w_diag = torch.zeros_like(xt)
            w_diag[..., :H // 4, :W // 4] = 1.0
            w_diag[..., :H // 4, 3 * W // 4:] = 1.0
            w_diag[..., 3 * H // 4:, :W // 4] = 1.0
            w_diag[..., 3 * H // 4:, 3 * W // 4:] = 1.0

            w_center = torch.zeros_like(xt)
            w_center[..., H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 1.0

            weights = [w_horiz, w_vert, w_diag, w_center]
            shift_names = ["horiz_half", "vert_half", "diag_half", "no_shift"]
        else:
            # 3-shift (the paper's setting): horiz handles edges (left+right+corners),
            # vert handles top/bottom, center handles interior
            w_horiz = torch.ones_like(xt)
            w_horiz[..., W // 4: 3 * W // 4] = 0.0
            w_vert = torch.ones_like(xt)
            w_vert[..., H // 4: 3 * H // 4, :] = 0.0
            w_center = torch.zeros_like(xt)
            w_center[..., H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 1.0
            weights = [w_horiz, w_vert, w_center]
            shift_names = ["horiz_half", "vert_half", "no_shift"]

    else:
        raise ValueError(
            f"tiling.direction={tiling_direction!r}; expected 'horizontal', "
            f"'vertical', or 'both'.")

    outputs = [denoiser.forward_shift(xt, t, name, scheduler) for name in shift_names]
    outputs = torch.stack(outputs)
    weights = torch.stack(weights)
    return (outputs * weights).sum(0) / weights.sum(0)


@timed_compute("sampling")
def sample_tileable(
    denoisers: Sequence[Optional[ImageDenoiser]],
    schedulers: Sequence[Scheduler],
    pyramid_processor: BasePyramidProcessor,
    sample_pyramid_shapes: Sequence[Tuple[int, ...]],
    tiling_config: TilingConfig,
    eta: float = 0.0,
    vae: Optional[FluxVAE] = None,
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
    intermediate_output_dir: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Sample a seamlessly tileable image, by averaging each step over circular shifts.

    Parameters
    ----------
    denoisers : list[ImageDenoiser]
        One per scale.
    schedulers : list[Scheduler]
        One per scale.
    pyramid_processor : BasePyramidProcessor
        Supplies the Laplacian consistency blend.
    sample_pyramid_shapes : list[tuple]
        Output (C, H, W) at each scale.
    tiling_config : TilingConfig
        Which seams to close, and how. ``num_shifts`` only affects "both" -- see
        ``blend_shifts``.
    eta : float
        Sampler stochasticity in [0, 1] (0 = deterministic).
    vae : FluxVAE or None
        None = the tensors here are pixels. Otherwise they are latents in this VAE's space,
        decoded to pixels before returning; the circular shifts are plain spatial rolls, so
        only the decode needs care.
    device : str or torch.device
        Device the sampling runs on.
    dtype : torch.dtype
        Precision the sampling runs in.
    intermediate_output_dir : str or None
        If set, save each scale's finished result here (the coarse-to-fine trace).

    Returns
    -------
    Tensor
        The tileable image as (C, H, W) in [0, 1].
    """
    num_scales = len(sample_pyramid_shapes)

    # Initialize noise at all scales
    xt = [scheduler.init_noise(shape, device, dtype)
          for scheduler, shape in zip(schedulers, sample_pyramid_shapes)]
    x_hat = [torch.zeros_like(noisy) for noisy in xt]

    # Coarse-to-fine sampling — all scales from noise
    for s in reversed(range(num_scales)):
        # Every index but the terminal at 0; the t=1 step lands there and ends the scale. That
        # step keeps the shift blend and skips the cross-scale one, so the result stays tileable.
        for t in reversed(range(1, schedulers[s].num_steps + 1)):
            x_hat[s] = blend_shifts(
                denoisers[s],
                schedulers[s],
                xt[s],
                t,
                tiling_config.direction,
                tiling_config.num_shifts,
            )

            epsilon_hat = schedulers[s].epsilon_from_x_hat(xt[s], x_hat[s], t)

            if s != num_scales - 1 and t != 1:
                x_hat[s] = pyramid_processor.blend_two_scale(x_hat[s], xt[s + 1])

            xt[s] = schedulers[s].step(x_hat[s], epsilon_hat, t, eta=eta)

        save_scale_result(xt[s], s, intermediate_output_dir, vae)

    # Result (decode from latent to pixels first when sampling in latent space)
    decoded = maybe_decode(xt[0], vae, tileable=tiling_config.periodic_decode)
    print(f"  seam error of the result: {seam_error(decoded, tiling_config.direction):.2f}/255"
          + (f" (periodic_decode={tiling_config.periodic_decode})"
             if vae is not None else ""))
    return unnormalize_to_zero_to_one(decoded).squeeze(0)
