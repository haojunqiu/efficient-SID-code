"""Structural analogy: image-based style transfer between a structure image and a style image.

Algorithm
---------
1. Initialize the coarsest scale: invert the *structure* image's coarsest scale to noise
   (``efficient_sid.inversion.invert_to_noise``) and reconstruct it with the *style* image's
   denoiser (``efficient_sid.inversion.reconstruct_from_noise``), which puts the style's
   texture on the structure's layout.
2. Sample coarse-to-fine from there with the style image's denoisers.
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch

from efficient_sid.inversion import invert_to_noise, reconstruct_from_noise
from efficient_sid.image_denoiser import ImageDenoiser
from efficient_sid.latent import FluxVAE, maybe_decode
from efficient_sid.pyramid import BasePyramidProcessor
from efficient_sid.scheduler import Scheduler
from efficient_sid.timing import timed_compute
from efficient_sid.utils import save_image, save_scale_result, unnormalize_to_zero_to_one


@timed_compute("sampling")
def sample_structural_analogy(
    style_denoisers: Sequence[Optional[ImageDenoiser]],
    schedulers: Sequence[Scheduler],
    pyramid_processor: BasePyramidProcessor,
    sample_pyramid_shapes: Sequence[Tuple[int, ...]],
    structure_coarsest_denoiser: ImageDenoiser,
    structure_coarsest_image: torch.Tensor,
    noise_level: Optional[float] = None,
    eta: float = 0.0,
    inverted_xt: Optional[torch.Tensor] = None,
    vae: Optional[FluxVAE] = None,
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
    intermediate_output_dir: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Sample the structure image's layout with the style image's patches: the structure's
    coarsest scale is inverted, then reconstructed and refined by the style denoisers.

    Parameters
    ----------
    style_denoisers : list[ImageDenoiser]
        One per scale, built from the style image.
    schedulers : list[Scheduler]
        One per scale; ``num_steps`` may differ between them.
    pyramid_processor : BasePyramidProcessor
        Supplies the Laplacian consistency blend.
    sample_pyramid_shapes : list[tuple]
        Output (C, H, W) at each scale, on the structure image's grid.
    structure_coarsest_denoiser : ImageDenoiser
        Built from the structure image's coarsest scale only, and used just for the inversion.
    structure_coarsest_image : Tensor
        (1, C, H, W) structure image's coarsest pyramid scale, in [-1, 1].
    noise_level : float or None
        The noise level to invert the structure image up to, and sample back down from --
        how much of the noisy image is noise, from 0 to 1. None inverts fully. Lower stays
        closer to the structure image, higher gives the style more freedom.
    eta : float
        Sampler stochasticity in [0, 1] (0 = deterministic).
    inverted_xt : Tensor or None
        A precomputed (1, C, H, W) inversion to start from, produced at this same
        ``noise_level`` with ``structure_coarsest_denoiser``. Skips the inversion.
    vae : FluxVAE or None
        None = the tensors here are pixels. Otherwise they are latents in this VAE's space,
        decoded to pixels before returning.
    device : str or torch.device
        Device the sampling runs on.
    dtype : torch.dtype
        Precision the sampling runs in.
    intermediate_output_dir : str or None
        If set, save the inverted coarsest scale and each scale's finished result here.

    Returns
    -------
    Tensor
        The analogy image as (C, H, W) in [0, 1].
    """
    num_scales = len(sample_pyramid_shapes)
    s_coarsest = num_scales - 1
    if intermediate_output_dir is not None:
        os.makedirs(intermediate_output_dir, exist_ok=True)

    # --- Phase 1: invert the structure image's coarsest scale to noise ---
    if inverted_xt is not None:
        xt_coarsest = inverted_xt.clone().to(dtype)
    else:
        xt_coarsest = invert_to_noise(
            structure_coarsest_denoiser,
            schedulers[s_coarsest],
            structure_coarsest_image,
            noise_level,
        )
    if intermediate_output_dir is not None:
        # Decoded first in latent space: this shows what the structure looks like at noise_level.
        save_image(
            unnormalize_to_zero_to_one(maybe_decode(xt_coarsest, vae)).squeeze(0),
            os.path.join(intermediate_output_dir, f"scale{s_coarsest}_inverted_noise.png"),
        )

    # --- Initialize all scales; finer scales start from pure noise ---
    xt = [scheduler.init_noise(shape, device, dtype)
          for scheduler, shape in zip(schedulers, sample_pyramid_shapes)]
    x_hat = [torch.zeros_like(noisy) for noisy in xt]

    # --- Phase 2a: reconstruct coarsest scale with the style denoiser ---
    xt[s_coarsest] = reconstruct_from_noise(
        style_denoisers[s_coarsest],
        schedulers[s_coarsest],
        xt_coarsest,
        noise_level=noise_level,
    ).to(dtype)
    x_hat[s_coarsest] = xt[s_coarsest]

    save_scale_result(xt[s_coarsest], s_coarsest, intermediate_output_dir, vae)

    # --- Phase 2b: coarse-to-fine sampling on finer scales (style denoisers only) ---
    for s in reversed(range(num_scales - 1)):
        # Every index but the terminal at 0; the t=1 step lands there and ends the scale.
        for t in reversed(range(1, schedulers[s].num_steps + 1)):
            x_hat[s] = style_denoisers[s](xt=xt[s], t=t, scheduler=schedulers[s])

            # The estimated noise comes from the estimated clean image, before blending.
            epsilon_hat = schedulers[s].epsilon_from_x_hat(xt[s], x_hat[s], t)

            if t != 1:   # no blend on the step onto the terminal
                x_hat[s] = pyramid_processor.blend_two_scale(x_hat[s], xt[s + 1])

            xt[s] = schedulers[s].step(x_hat[s], epsilon_hat, t, eta=eta)

        save_scale_result(xt[s], s, intermediate_output_dir, vae)

    return unnormalize_to_zero_to_one(maybe_decode(xt[0], vae)).squeeze(0)
