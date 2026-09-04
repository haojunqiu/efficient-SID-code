"""Single-scale DDIM inversion: invert a real image to noise, and reconstruct back.

For applications that start from a real image instead of pure noise -- structural analogy
inverts one pyramid scale, text style inverts the whole image.

``noise_level`` says how far to invert as a fraction rather than a step index, so it carries
across ``num_steps``. ``Scheduler.index_for_noise_level`` resolves it to a step.
"""

from typing import Optional

import torch

from .image_denoiser import ImageDenoiser
from .scheduler import Scheduler
from .timing import timed


@timed("inversion (incl. its denoiser calls)", nested=True)
def invert_to_noise(
    denoiser: ImageDenoiser,
    scheduler: Scheduler,
    image: torch.Tensor,
    noise_level: Optional[float] = None,
) -> torch.Tensor:
    """DDIM-invert a clean image into a noisy latent (data -> noise, eta=0).

    image: (1, C, H, W)
    noise_level: the fraction to stop at; None inverts fully.
    Returns: (1, C, H, W) noisy latent, sitting at the index ``noise_level`` resolves to.
    """
    stop_idx = scheduler.index_for_noise_level(noise_level)

    # Index 1 is the least noisy evaluation point, where the walk upward starts.
    xt = scheduler.alphas[1] * image + scheduler.sigmas[1] * torch.randn_like(image)
    for t in range(1, stop_idx):
        x_hat = denoiser(xt, t, scheduler)
        epsilon_hat = scheduler.epsilon_from_x_hat(xt, x_hat, t)
        # Inversion goes one step noisier, so t -> t + 1.
        xt = scheduler.step(x_hat, epsilon_hat, t, target_t=t + 1)
    return xt


def reconstruct_from_noise(
    denoiser: ImageDenoiser,
    scheduler: Scheduler,
    xt: torch.Tensor,
    noise_level: Optional[float] = None,
) -> torch.Tensor:
    """DDIM-reconstruct an image from a noisy latent (noise -> data, eta=0).

    denoiser may differ from the one used to invert (e.g. a different image's denoiser, for
    structural analogy / style transfer).

    noise_level: where ``xt`` sits, in the same [0, 1] convention as ``invert_to_noise``. None =
        maximum noise, i.e. a full reconstruction.
    Returns: (1, C, H, W) reconstructed image.
    """
    start_idx = scheduler.index_for_noise_level(noise_level)

    # The t=1 step lands on the terminal, so xt is the clean image when the loop ends.
    for t in reversed(range(1, start_idx + 1)):
        x_hat = denoiser(xt, t, scheduler)
        epsilon_hat = scheduler.epsilon_from_x_hat(xt, x_hat, t)
        xt = scheduler.step(x_hat, epsilon_hat, t)

    return xt
