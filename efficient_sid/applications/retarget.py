"""Retargeting: generate an image at a different resolution using the single-image prior.

The patch dataset comes from the original image, but the sampling canvas has the target
shape -- so the output is built out of the input's patches, rearranged to fit a new aspect
ratio, rather than resized.

Algorithm
---------
1. Initialize the coarsest scale from the resized image, which fixes the overall layout at the
   target aspect ratio.
2. Sample coarse-to-fine from there with the original image's denoisers, so the result keeps
   the input's patch distribution.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch

from efficient_sid.image_denoiser import ImageDenoiser
from efficient_sid.latent import FluxVAE, maybe_decode
from efficient_sid.pyramid import BasePyramidProcessor
from efficient_sid.scheduler import Scheduler
from efficient_sid.timing import timed_compute
from efficient_sid.utils import save_scale_result, unnormalize_to_zero_to_one


@timed_compute("sampling")
def sample_retargeting(
    denoisers: Sequence[Optional[ImageDenoiser]],
    schedulers: Sequence[Scheduler],
    pyramid_processor: BasePyramidProcessor,
    sample_pyramid_shapes: Sequence[Tuple[int, ...]],
    resized_pyramid: Sequence[torch.Tensor],
    eta: float = 0.0,
    vae: Optional[FluxVAE] = None,
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
    intermediate_output_dir: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Sample the input's content onto a different shape -- rearranged, not resized.

    The coarsest scale is initialized from the resized image, which pins the layout to the
    target shape; the finer scales then denoise coarse-to-fine with the original image's
    patches, which is where the texture comes from.

    Parameters
    ----------
    denoisers : list[ImageDenoiser]
        One per scale, built from the *original* image's patches. The coarsest entry is None --
        that scale is never denoised.
    schedulers : list[Scheduler]
        One per scale.
    pyramid_processor : BasePyramidProcessor
        Supplies the Laplacian consistency blend.
    sample_pyramid_shapes : list[tuple]
        Target (C, H, W) at each scale.
    resized_pyramid : list[Tensor]
        Gaussian pyramid of the original image at the target resolution, each (C, H, W) in
        [-1, 1]. Its coarsest scale is what the sampling starts from.
    eta : float
        Sampler stochasticity in [0, 1] (0 = deterministic).
    vae : FluxVAE or None
        None = the tensors here are pixels. Otherwise they are latents in this VAE's space,
        decoded to pixels before returning.
    device : str or torch.device
        Device the sampling runs on.
    dtype : torch.dtype
        Precision the sampling runs in.
    intermediate_output_dir : str or None
        If set, save each scale's finished result here (the coarse-to-fine trace).

    Returns
    -------
    Tensor
        The retargeted image as (C, H, W) in [0, 1].
    """
    num_scales = len(sample_pyramid_shapes)

    # --- Coarsest scale: taken directly from the resized image, never denoised. This is what
    # pins the layout to the target aspect ratio; the finer scales then supply the texture. ---
    s_coarsest = num_scales - 1

    xt = [None if s == s_coarsest else scheduler.init_noise(shape, device, dtype)
          for s, (scheduler, shape) in enumerate(zip(
              schedulers,
              sample_pyramid_shapes,
          ))]
    xt[s_coarsest] = resized_pyramid[s_coarsest].unsqueeze(0).to(dtype).to(device)

    x_hat = [torch.zeros_like(noisy) for noisy in xt]
    x_hat[s_coarsest] = xt[s_coarsest].clone()

    # --- Coarse-to-fine sampling (skip coarsest) ---
    for s in reversed(range(num_scales - 1)):
        # Every index but the terminal at 0; the t=1 step lands there and ends the scale.
        for t in reversed(range(1, schedulers[s].num_steps + 1)):
            x_hat[s] = denoisers[s](xt=xt[s], t=t, scheduler=schedulers[s])

            epsilon_hat = schedulers[s].epsilon_from_x_hat(xt[s], x_hat[s], t)

            if t != 1:   # no blend on the step onto the terminal
                x_hat[s] = pyramid_processor.blend_two_scale(x_hat[s], xt[s + 1])

            xt[s] = schedulers[s].step(x_hat[s], epsilon_hat, t, eta=eta)

        save_scale_result(xt[s], s, intermediate_output_dir, vae)

    return unnormalize_to_zero_to_one(maybe_decode(xt[0], vae)).squeeze(0)
