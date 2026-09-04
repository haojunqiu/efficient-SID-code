"""Text-driven style transfer: restyle a single image toward a text prompt with CLIP guidance.

Algorithm
---------
1. Invert the content image to ``noise_level`` -- the start point t' -- with its own
   denoiser (``efficient_sid.inversion.invert_to_noise``).
2. DDIM-reconstruct with the same denoiser, nudging the clean estimate toward the prompt at
   every step with CLIP guidance (``clip_guidance_step``).

There is no style image: the content image is both what gets inverted and the patch prior it is
reconstructed with. Single-scale, so there is no pyramid and no ``num_scales`` knob.
"""

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from efficient_sid.clip_extractor import ClipExtractor, get_augmentations_template
from efficient_sid.image_denoiser import ImageDenoiser
from efficient_sid.inversion import invert_to_noise
from efficient_sid.scheduler import Scheduler
from efficient_sid.timing import timed_compute
from efficient_sid.utils import save_image, unnormalize_to_zero_to_one

# -------------------------------------------------------------------------
# Thresholded CLIP gradient: soft-threshold the gradient and mask all but the top-f fraction
# -------------------------------------------------------------------------

@dataclass
class ClipGuidanceConfig:
    """How hard each diffusion step is pushed toward the prompt."""
    strength: float = 1.0        # gamma, time-scaled per step by sigmas[t]/sigmas[t']
    llambda: float = 0.1         # weight on the new estimate; lower keeps more momentum
    fill_factor: float = 0.5     # fraction of pixels guided each step
    iters: int = 1               # guidance refinements per diffusion step


# ---------------------------------------------------------------------------
# Thresholded CLIP gradient: soft-threshold the gradient and mask all but the top-f fraction
# ---------------------------------------------------------------------------

def thresholded_grad(grad: torch.Tensor, quantile: float = 0.8) -> torch.Tensor:
    """Soft-threshold a (1,C,H,W) CLIP gradient: keep only the top-(1-quantile) energy pixels.

    Returns (sparse_grad, mask). mask marks the retained pixel positions.
    """
    grad_energy = torch.norm(grad, dim=1)                       # (B,H,W)
    flat = grad_energy.reshape(grad_energy.shape[0], -1)
    # torch.quantile only accepts float/double, so compute the threshold in fp32 and cast back --
    # otherwise a bfloat16 denoiser would make this raise.
    q = torch.quantile(flat.float(), q=quantile, dim=1, interpolation="nearest")
    q = q.to(grad_energy.dtype)[:, None, None]
    diff = grad_energy - q
    mask = (diff > 0)[:, None, :, :]
    diff_clamp = torch.clamp(diff, min=0)[:, None, :, :]
    unit = grad / grad_energy[:, None, :, :]
    unit[torch.isnan(unit)] = 0
    sparse_grad = diff_clamp * unit
    return sparse_grad, mask


def clip_guidance_step(
    x_hat: torch.Tensor,
    x_hat_prev: Optional[torch.Tensor],
    clip_extractor: ClipExtractor,
    text_embed: torch.Tensor,
    guidance_strength_t: float,
    mask_quantile: float,
    momentum_lambda: float,
) -> torch.Tensor:
    """Apply one CLIP-guided refinement to a clean-image estimate.

    Applies momentum from the previous step's estimate, then takes a single masked,
    self-normalized step along the CLIP text-similarity gradient and clamps back to [-1, 1].

    x_hat, x_hat_prev : (1, C, H, W) clean estimates in [-1, 1]; ``x_hat_prev`` may be None
        on the first step (no momentum then).
    text_embed : precomputed CLIP text embedding(s) for the prompt.
    guidance_strength_t : the (time-scaled) step size for this diffusion step.
    mask_quantile : quantile passed to ``thresholded_grad`` (= 1 - fill_factor).
    momentum_lambda : momentum weight on the new estimate vs. the previous one.
    Returns the updated (1, C, H, W) estimate.
    """
    x_hat_mom = (x_hat if x_hat_prev is None
                 else (1 - momentum_lambda) * x_hat_prev + momentum_lambda * x_hat)

    # Gradient of CLIP text-similarity is taken at the momentum-blended estimate.
    x_guided = x_hat_mom.detach().clone().requires_grad_(True)
    # CLIP expects images in [0, 1]
    score = -clip_extractor.calculate_clip_loss(
        unnormalize_to_zero_to_one(x_guided),
        text_embed,
    )
    grad = torch.autograd.grad(score, x_guided)[0]

    grad, mask = thresholded_grad(grad=grad, quantile=mask_quantile)
    mask = mask.float()

    # Self-normalize the step to the masked image magnitude so a single strength works across
    # images/steps (norms computed over channels+spatial dims, per sample).
    norm_img = torch.linalg.vector_norm(x_guided.detach() * mask, dim=(1, 2, 3), keepdim=True)
    norm_grad = torch.linalg.vector_norm(grad.detach() * mask, dim=(1, 2, 3), keepdim=True)
    division_norm = norm_img / norm_grad

    x_hat = x_hat_mom + guidance_strength_t * division_norm * grad * mask
    return x_hat.clamp(-1.0, 1.0)



@timed_compute("sampling")
def sample_text_style(
    content_denoiser: ImageDenoiser,
    scheduler: Scheduler,
    clip_extractor: ClipExtractor,
    content_image: torch.Tensor,
    text: str,
    clip_config: ClipGuidanceConfig,
    noise_level: float = 0.3,
    eta: float = 0.0,
    inverted_xt: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    intermediate_output_dir: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Sample a restyled image guided by a text prompt: the content image is inverted, then
    reconstructed with a CLIP gradient applied at every step.

    Parameters
    ----------
    content_denoiser : ImageDenoiser
        Built from the content image, and used for both the inversion and the reconstruction.
    scheduler : Scheduler
        The single scheduler this application samples on.
    clip_extractor : ClipExtractor
        Augmented CLIP image/text embedder supplying ``get_text_embedding`` and
        ``calculate_clip_loss``.
    content_image : Tensor
        (1, C, H, W) in [-1, 1] -- the image being restyled.
    text : str
        The prompt to guide sampling toward, e.g. "Van Gogh style".
    clip_config : ClipGuidanceConfig
    noise_level : float
        The noise level to invert the content image up to, and sample back down from --
        how much of the noisy image is noise, from 0 to 1. Lower stays closer to the content,
        higher gives the prompt more freedom.
    eta : float
        Sampler stochasticity in [0, 1] (0 = deterministic).
    inverted_xt : Tensor or None
        A precomputed (1, C, H, W) inversion to start from, produced at this same
        ``noise_level`` with this image's denoiser. Skips the inversion.
    dtype : torch.dtype
        Precision the sampling runs in.
    intermediate_output_dir : str or None
        If set, save the inverted latent and each step's guided result here.

    Returns
    -------
    Tensor
        The restyled image as (C, H, W) in [0, 1].
    """
    if intermediate_output_dir is not None:
        os.makedirs(intermediate_output_dir, exist_ok=True)

    mask_quantile = 1.0 - clip_config.fill_factor

    # Precompute the prompt's CLIP text embedding once (high-res augmentation templates).
    text_embed = clip_extractor.get_text_embedding(
        text,
        template=get_augmentations_template("hr"),
    )

    # Where the inverted latent sits: the index invert_to_noise stopped at, and so the index the
    # loop starts from. start_sigma is its noise level there, used to time-scale the guidance.
    start_idx = scheduler.index_for_noise_level(noise_level)
    start_sigma = scheduler.sigmas[start_idx]

    def apply_guidance(
        x_hat: torch.Tensor,
        t: int,
        x_hat_prev: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Nudge one clean estimate toward the prompt: the CLIP half of a step.

        The result is the estimate the sampler goes on to use, so epsilon, the DDIM step and the
        t=0 result all see the guided value.
        """
        # Fade the strength out with the scheduler so late steps barely move.
        strength_t = (scheduler.sigmas[t] / start_sigma) * clip_config.strength
        for _ in range(clip_config.iters):
            x_hat = clip_guidance_step(
                x_hat=x_hat,
                x_hat_prev=x_hat_prev,
                clip_extractor=clip_extractor,
                text_embed=text_embed,
                guidance_strength_t=strength_t,
                mask_quantile=mask_quantile,
                momentum_lambda=clip_config.llambda,
            ).to(dtype)
        return x_hat

    def save_step_result(x: torch.Tensor, t: int) -> None:
        """Save one step's result. A no-op unless intermediates were asked for, like
        ``utils.save_scale_result`` for the other applications."""
        if intermediate_output_dir is None:
            return
        save_image(
            unnormalize_to_zero_to_one(x).squeeze(0),
            os.path.join(intermediate_output_dir, f"t={t}_result.png"),
        )

    # --- Init by inverting the content image to ``noise_level`` ---
    if inverted_xt is not None:
        xt = inverted_xt.clone().to(dtype)
    else:
        xt = invert_to_noise(
            content_denoiser,
            scheduler,
            content_image,
            noise_level,
        ).to(dtype)
    if intermediate_output_dir is not None:
        # What inversion preserved of the content image at this ``noise_level``.
        save_image(
            unnormalize_to_zero_to_one(xt).squeeze(0),
            os.path.join(intermediate_output_dir, "inverted_noise.png"),
        )

    # --- Reconstruct t' -> 0, injecting CLIP guidance at every step ---
    x_hat_prev = None
    # The t=1 step lands on the terminal, so xt is the guided clean image when the loop ends.
    for t in reversed(range(1, start_idx + 1)):
        x_hat = content_denoiser(xt=xt, t=t, scheduler=scheduler)
        x_hat = apply_guidance(x_hat, t, x_hat_prev)
        x_hat_prev = x_hat.detach()

        epsilon_hat = scheduler.epsilon_from_x_hat(xt, x_hat, t)
        xt = scheduler.step(x_hat, epsilon_hat, t, eta=eta)
        save_step_result(xt, t)

    return unnormalize_to_zero_to_one(xt).squeeze(0)
