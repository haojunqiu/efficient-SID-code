"""Symmetric generation: produce horizontally or vertically symmetric images.

Algorithm
---------
Standard coarse-to-fine sampling with one addition: after every denoiser call, mirror one
half of the image onto the other.

Every scale starts from noise, so the result is a random symmetric image drawn from the
input's patch distribution rather than a symmetrized copy of the input.

In latent mode the decoder is not mirror-equivariant, so the mirror is applied once more
after decoding (a no-op in pixel space).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from efficient_sid.image_denoiser import ImageDenoiser
from efficient_sid.latent import FluxVAE, maybe_decode
from efficient_sid.pyramid import BasePyramidProcessor
from efficient_sid.scheduler import Scheduler
from efficient_sid.timing import timed_compute
from efficient_sid.utils import save_scale_result, unnormalize_to_zero_to_one


@dataclass
class SymmetryConfig:
    """Which axis to mirror across, and which half survives."""
    axis: str = "horizontal"         # "horizontal" or "vertical"
    mirror_source: str = "left"      # "left"/"right" for horizontal, "top"/"bottom" for vertical


def mirror_asymmetry(img: torch.Tensor, symmetry: str) -> torch.Tensor:
    """Measure mean |img - mirror(img)| in 0-255 units — 0 iff the image is exactly symmetric.

    On the result ``sample_symmetric`` returns this reads 0 in both spaces: sampling enforces
    symmetry on whatever tensor it is given, and a latent run applies ``sym_fn`` once more after
    decoding. It is a guard rather than a reported approximation — a nonzero value means something
    after that last ``sym_fn`` broke the mirror. To see the decoder's own mirror error, call this
    on the decode output *before* the pixel-space fix-up.
    """
    flipped = img.flip(-1) if symmetry == "horizontal" else img.flip(-2)
    return (img - flipped).abs().mean().item() * 255.0 / 2.0


def symmetrize_image_horizontal(img: torch.Tensor, source: str = 'left') -> torch.Tensor:
    """Enforce horizontal symmetry by mirroring one half onto the other.

    Parameters
    ----------
    img : Tensor (B, C, H, W) or (C, H, W)
        Input image in any value range.
    source : str
        "left" — keep the left half, mirror to right.
        "right" — keep the right half, mirror to left.

    Returns
    -------
    Tensor
        The symmetrized image, same shape and range as ``img``.
    """
    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    W = img.shape[-1]
    mid = W // 2

    if source == "left":
        left = img[..., :mid]
        right = left.flip(-1)
        # If odd width, keep center column from left
        if W % 2 == 1:
            out = torch.cat([left, img[..., mid:mid+1], right], dim=-1)
        else:
            out = torch.cat([left, right], dim=-1)
    elif source == "right":
        right = img[..., -mid:] if mid > 0 else img[..., :0]
        left = right.flip(-1)
        if W % 2 == 1:
            out = torch.cat([left, img[..., mid:mid+1], right], dim=-1)
        else:
            out = torch.cat([left, right], dim=-1)
    else:
        raise ValueError(
            f"symmetry.mirror_source={source!r}; expected 'left' or 'right' "
            f"for a horizontal axis.")

    if squeeze:
        out = out.squeeze(0)
    return out


def symmetrize_image_vertical(img: torch.Tensor, source: str = 'top') -> torch.Tensor:
    """Enforce vertical symmetry by mirroring one half onto the other.

    Parameters
    ----------
    img : Tensor (B, C, H, W) or (C, H, W)
        Input image in any value range.
    source : str
        "top" — keep the top half, mirror to bottom.
        "bottom" — keep the bottom half, mirror to top.

    Returns
    -------
    Tensor
        The symmetrized image, same shape and range as ``img``.
    """
    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    H = img.shape[-2]
    mid = H // 2

    if source == "top":
        top = img[..., :mid, :]
        bottom = top.flip(-2)
        if H % 2 == 1:
            out = torch.cat([top, img[..., mid:mid+1, :], bottom], dim=-2)
        else:
            out = torch.cat([top, bottom], dim=-2)
    elif source == "bottom":
        bottom = img[..., -mid:, :] if mid > 0 else img[..., :0, :]
        top = bottom.flip(-2)
        if H % 2 == 1:
            out = torch.cat([top, img[..., mid:mid+1, :], bottom], dim=-2)
        else:
            out = torch.cat([top, bottom], dim=-2)
    else:
        raise ValueError(
            f"symmetry.mirror_source={source!r}; expected 'top' or 'bottom' "
            f"for a vertical axis.")

    if squeeze:
        out = out.squeeze(0)
    return out


def _get_sym_fn(symmetry: str, mirror_source: str) -> Callable:
    """Return the appropriate symmetrization function."""
    if symmetry == "horizontal":
        return lambda img: symmetrize_image_horizontal(img, source=mirror_source)
    elif symmetry == "vertical":
        return lambda img: symmetrize_image_vertical(img, source=mirror_source)
    else:
        raise ValueError(
            f"symmetry.axis={symmetry!r}; expected 'horizontal' or 'vertical'.")


@timed_compute("sampling")
def sample_symmetric(
    denoisers: Sequence[Optional[ImageDenoiser]],
    schedulers: Sequence[Scheduler],
    pyramid_processor: BasePyramidProcessor,
    sample_pyramid_shapes: Sequence[Tuple[int, ...]],
    symmetry_config: SymmetryConfig,
    eta: float = 0.0,
    vae: Optional[FluxVAE] = None,
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
    intermediate_output_dir: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Sample an image symmetric about one axis, by mirroring after every denoiser call.

    Parameters
    ----------
    denoisers : list[ImageDenoiser]
        One per scale, built from the original image's patches.
    schedulers : list[Scheduler]
        One per scale.
    pyramid_processor : BasePyramidProcessor
        Supplies the Laplacian consistency blend.
    sample_pyramid_shapes : list[tuple]
        Output (C, H, W) at each scale.
    symmetry_config : SymmetryConfig
        Which axis to mirror across, and which half survives.
    eta : float
        Sampler stochasticity in [0, 1] (0 = deterministic).
    vae : FluxVAE or None
        None = the tensors here are pixels. Otherwise they are latents in this VAE's space,
        decoded to pixels before returning. The decoder is not mirror-equivariant, so the
        mirror is applied once more in pixel space to make the result exact.
    device : str or torch.device
        Device the sampling runs on.
    dtype : torch.dtype
        Precision the sampling runs in.
    intermediate_output_dir : str or None
        If set, save each scale's finished result here (the coarse-to-fine trace).

    Returns
    -------
    Tensor
        The symmetric image as (C, H, W) in [0, 1].
    """
    sym_fn = _get_sym_fn(symmetry_config.axis, symmetry_config.mirror_source)

    num_scales = len(sample_pyramid_shapes)

    xt = [scheduler.init_noise(shape, device, dtype)
          for scheduler, shape in zip(schedulers, sample_pyramid_shapes)]

    x_hat = [torch.zeros_like(noisy) for noisy in xt]

    # --- Coarse-to-fine sampling ---
    for s in reversed(range(num_scales)):
        # Every index but the terminal at 0; the t=1 step lands there and ends the scale. That
        # step applies sym_fn and skips the blend, so the finished latent is exactly symmetric.
        for t in reversed(range(1, schedulers[s].num_steps + 1)):
            x_hat[s] = denoisers[s](xt=xt[s], t=t, scheduler=schedulers[s])
            x_hat[s] = sym_fn(x_hat[s])

            epsilon_hat = schedulers[s].epsilon_from_x_hat(xt[s], x_hat[s], t)

            if s != num_scales - 1 and t != 1:
                x_hat[s] = pyramid_processor.blend_two_scale(x_hat[s], xt[s + 1])

            xt[s] = schedulers[s].step(x_hat[s], epsilon_hat, t, eta=eta)

        save_scale_result(xt[s], s, intermediate_output_dir, vae)

    # --- Save result (decode from latent to pixels first when sampling in latent space) ---
    result = maybe_decode(xt[0], vae)
    if vae is not None:
        # Decoder is not mirror-equivariant, so a symmetric latent decodes only ~symmetric;
        # re-apply the mirror in pixel space to restore it exactly.
        result = sym_fn(result)
    print(f"  mirror asymmetry of the result: {mirror_asymmetry(result, symmetry_config.axis):.2f}/255")
    return unnormalize_to_zero_to_one(result).squeeze(0)
