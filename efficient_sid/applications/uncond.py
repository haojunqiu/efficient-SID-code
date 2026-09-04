"""Unconditional generation: sample a new image from one image's patches.

Algorithm
---------
Each pyramid scale is sampled from noise, coarsest first. After every denoiser call the clean
estimate is Laplacian-blended with the coarser scale already finished, which holds the finer
scale to the layout the coarser one settled. The finest scale is the result.

This is the base the other five modify: retarget fixes the coarsest scale from a resized image,
symmetric mirrors after every denoiser call, tileable blends over circular shifts, structural
analogy reconstructs an inverted structure scale with the style denoiser, and text style uses a
single scale, with CLIP guidance.
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
def sample_uncond(
    denoisers: Sequence[Optional[ImageDenoiser]],
    schedulers: Sequence[Scheduler],
    pyramid_processor: BasePyramidProcessor,
    sample_pyramid_shapes: Sequence[Tuple[int, ...]],
    eta: float = 0.0,
    vae: Optional[FluxVAE] = None,
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
    intermediate_output_dir: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Sample an image from the single-image prior, coarse scale to fine.

    Each scale runs a full DDIM trajectory; after every denoiser call the clean estimate is
    Laplacian-blended with the finished coarser scale above it, which keeps the scales
    consistent. The finest scale is the result.

    Parameters
    ----------
    denoisers : list[ImageDenoiser]
        One per scale, coarsest last.
    schedulers : list[Scheduler]
        One per scale.
    pyramid_processor : BasePyramidProcessor
        Supplies the Laplacian consistency blend.
    sample_pyramid_shapes : list[tuple]
        Output (C, H, W) at each scale. Need not match the input image's shape -- pass a
        different one to retarget.
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
        The image as (C, H, W) in [0, 1], decoded to pixels when sampling in latent space.
    """
    num_scales = len(sample_pyramid_shapes)

    xt = [scheduler.init_noise(shape, device, dtype)
          for scheduler, shape in zip(schedulers, sample_pyramid_shapes)]
    x_hat = [torch.zeros_like(noisy) for noisy in xt]

    for s in reversed(range(num_scales)):
        # Every index but the terminal at 0; the t=1 step lands there and ends the scale.
        for t in reversed(range(1, schedulers[s].num_steps + 1)):
            x_hat[s] = denoisers[s](xt=xt[s], t=t, scheduler=schedulers[s])

            # The estimated noise comes from the estimated clean image, before blending.
            epsilon_hat = schedulers[s].epsilon_from_x_hat(xt[s], x_hat[s], t)

            if s != num_scales - 1 and t != 1:   # no blend on the step onto the terminal
                x_hat[s] = pyramid_processor.blend_two_scale(x_hat[s], xt[s + 1])

            xt[s] = schedulers[s].step(x_hat[s], epsilon_hat, t, eta=eta)

        save_scale_result(xt[s], s, intermediate_output_dir, vae)

    return unnormalize_to_zero_to_one(maybe_decode(xt[0], vae)).squeeze(0)
