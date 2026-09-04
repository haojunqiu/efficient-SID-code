"""Noise schedulers (flow_linear, ve_edm/Karras) and how to move along their schedules.

A scheduler holds two 1-D tensors ``(alphas, sigmas)``: a clean image ``x`` and its noise ``eps``
combine as ``x_t = alpha_t * x + sigma_t * eps`` at each step t. ``Scheduler`` owns that pair and the
operations that act on it; one subclass per scheduler type supplies the tensors, and
``make_scheduler`` picks the subclass from the config's ``type``.

    x_hat = denoiser(x_t, t)
    eps   = scheduler.epsilon_from_x_hat(x_t, x_hat, t)   # noise implied by x_hat
    x_t   = scheduler.step(x_hat, eps, t, eta)            # step to t-1
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from omegaconf import MISSING
from torch import nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """What every scheduler type has. One subclass per type carries that type's own settings."""
    type: str = MISSING
    #: scalar, or one entry per pyramid scale. Untyped because OmegaConf cannot express
    #: `Union[int, List[int]]` ("Unions of containers are not supported"); ``build.resolve``
    #: expands it.
    num_steps: Any = 10
    sigma_min: float = 1e-4
    #: how much of each step's noise is drawn fresh rather than reused; 0 is the
    #: deterministic step, 1 re-noises from scratch. See ``Scheduler.step``.
    eta: float = 0.0


@dataclass
class FlowLinearConfig(SchedulerConfig):
    """Linear alpha and sigma between 0 and 1. sigma_max is fixed at 1 and there is no rho."""
    type: str = "flow_linear"


@dataclass
class VeEdmConfig(SchedulerConfig):
    """Variance-exploding schedule of Karras et al. (EDM): alpha stays 1, only sigma grows."""
    type: str = "ve_edm"
    sigma_min: float = 0.002        # EDM's own floor, not flow_linear's
    sigma_max: float = 80.0         # EDM's own ceiling; flow_linear fixes its own at 1
    rho: float = 7.0                # step spacing; larger packs more steps at low noise


class Scheduler(nn.Module):
    """A noise scheduler: the ``(alphas, sigmas)`` pair and the operations that move along it.

    Index 0 is the LEAST noisy step and the last index the most, so ``sigmas[-1]`` is sigma_max;
    the index *is* the timestep. Index 0 itself is the trajectory's terminal,
    ``(alpha, sigma) = (1, 0)`` -- a fully clean image.
    ``step`` lands on it from t=1 and it ends every sampling loop, but the denoiser is NEVER
    evaluated there: the denoisers' kernel scales by ``1/sigma^2``, which is infinite at sigma=0 and
    would make their softmax NaN. Indices 1..num_steps are the evaluation points, so
    ``sigmas[1]`` is the smallest sigma the denoiser is ever evaluated at and
    ``len(alphas) == num_steps + 1``.

    An ``nn.Module`` so the tensors are registered buffers and follow ``.to()``/``.cuda()``.
    """

    def __init__(self, alphas: torch.Tensor, sigmas: torch.Tensor) -> None:
        super().__init__()
        # The terminal (1, 0) goes at index 0, below the least noisy evaluation point, so `step`
        # can land on it by index like any other destination.
        one = torch.ones(1, dtype=alphas.dtype, device=alphas.device)
        self.register_buffer('alphas', torch.cat([one, alphas]).contiguous())
        self.register_buffer('sigmas', torch.cat([torch.zeros_like(one), sigmas]).contiguous())

    def __len__(self) -> int:
        return self.num_steps

    @property
    def num_steps(self) -> int:
        """How many points the denoiser is evaluated at: every index except the terminal at 0."""
        return len(self.alphas) - 1

    @property
    def init_noise_sigma(self) -> torch.Tensor:
        """Std of x_T, the starting point of a from-scratch sample.

        x_T = alphas[-1] * x + sigmas[-1] * eps, so this is sigma_max. Exact under flow_linear,
        where alphas[-1] is 0; on a variance-exploding schedule it drops a sigma_data term that is
        negligible while sigma_max >> sigma_data.
        """
        return self.sigmas[-1]

    @property
    def noise_amplitude_fraction(self) -> torch.Tensor:
        """Per-step ``sigma / (alpha + sigma)``: how much of x_t is noise.

        Scheduler-invariant, so one ``noise_level`` means the same signal-to-noise mix on any
        scheduler. Under flow_linear ``alpha + sigma`` is 1, so this is just sigma.
        """
        return self.sigmas / (self.alphas + self.sigmas)

    def index_for_noise_level(self, noise_level: float) -> int:
        """The step index whose noise level is closest to ``noise_level``; None is full noise.

        ``noise_level`` is a ``noise_amplitude_fraction``: under flow_linear it is exactly sigma,
        since alpha + sigma = 1 at every step, which is what the shipped values are tuned against
        and what the paper used. On another schedule it need not be -- a variance-exploding one,
        for instance.

        Index 0 is never returned: a ``noise_level`` below the scheduler's smallest fraction snaps
        to index 1, the least noisy step the denoiser can still be evaluated at.
        """
        if noise_level is None:
            return len(self.sigmas) - 1
        idx = torch.argmin(torch.abs(self.noise_amplitude_fraction - noise_level)).item()
        return max(idx, 1)

    def epsilon_from_x_hat(self, xt: torch.Tensor, x_hat: torch.Tensor, t: int) -> torch.Tensor:
        """Return the noise implied by a clean-image estimate: invert x_t = alpha*x_hat + sigma*eps."""
        # At the terminal xt == x_hat and sigma == 0, so the inversion is 0/0 and every eps
        # satisfies it.
        assert t != 0, "no epsilon at the terminal: sigma = 0 makes it 0/0"
        return (xt - self.alphas[t] * x_hat) / self.sigmas[t]

    def step(
        self,
        x_hat: torch.Tensor,
        epsilon_hat: torch.Tensor,
        t: int,
        eta: float = 0.0,
        target_t: Optional[int] = None,
    ) -> torch.Tensor:
        """Step to a target noise level, from a clean estimate and the noise it implies.

        ``t`` is the current noise level. The update lands on ``t - 1``, the sampling direction,
        or on ``target_t`` -- inversion passes ``t + 1`` to walk up. It reads only the
        destination's ``alpha`` and ``sigma``, so ``t`` is unused whenever ``target_t`` is given.

        ``eta`` in [0, 1] sets how much of the noise is drawn fresh rather than reused from
        ``epsilon_hat``. eta=0 is the deterministic DDIM step (a Euler step on the velocity field
        under flow_linear); eta=1 discards ``epsilon_hat`` entirely and re-noises from scratch.
        Note this is NOT the eta of the DDIM paper, whose eta=1 is DDPM ancestral sampling -- but
        every value here is valid: the noise variance stays sigma^2 throughout, since
        (1 - eta^2) + eta^2 = 1, so the marginal is preserved.

        The update is Eq. (5) of the paper.
        """
        assert 0.0 <= eta <= 1.0
        if target_t is None:
            target_t = t - 1
        # alphas[0] = 1 and sigmas[0] = 0, so landing on the terminal gives x_hat itself, with no
        # fresh noise drawn at any eta.
        if target_t == 0:
            return x_hat
        alpha, sigma = self.alphas[target_t], self.sigmas[target_t]
        if eta == 0.0:
            return alpha * x_hat + sigma * epsilon_hat
        fresh_noise = torch.randn_like(x_hat)
        sto_coef = eta * sigma
        det_coef = (sigma ** 2 - sto_coef ** 2) ** 0.5
        return alpha * x_hat + det_coef * epsilon_hat + sto_coef * fresh_noise

    def init_noise(
        self,
        shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Draw one scale's from-scratch x_T: noise at this scheduler's sigma_max."""
        noise = torch.randn(1, *shape, device=device, dtype=dtype).contiguous()
        return self.init_noise_sigma.to(device=device, dtype=dtype) * noise


class FlowLinearScheduler(Scheduler):
    """The default scheduler, selected by ``scheduler.type: "flow_linear"``.

    alpha falls linearly from ~1 to 0 and sigma = 1 - alpha, so x_t interpolates straight from the
    image to pure noise -- the flow-matching / rectified-flow parameterization. Since alpha + sigma
    is 1 at every step, ``sigma_min`` is exactly this scheduler's smallest sigma and sigma_max is
    fixed at 1.
    """

    #: The config class this scheduler is written against; ``config.load`` validates a written
    #: block with it.
    config_cls = FlowLinearConfig

    def __init__(self, config: FlowLinearConfig, device: Optional[torch.device] = None) -> None:
        alphas = torch.linspace(
            1 - config.sigma_min,
            0.0,
            config.num_steps,
            device=device,
            dtype=torch.float32,
        ).contiguous()
        super().__init__(alphas, 1.0 - alphas)


class VEEDMScheduler(Scheduler):
    """The variance-exploding EDM scheduler of Karras et al., selected by
    ``scheduler.type: "ve_edm"``.

    Variance-exploding: alpha stays 1 and only sigma grows, so noise is *added* to an
    undiminished image rather than mixed with it. The sigmas are spaced by the rho-warped rule
    from EDM (rho=7 packs more steps at low noise, where the estimate changes fastest).
    """

    config_cls = VeEdmConfig

    def __init__(self, config: VeEdmConfig, device: Optional[torch.device] = None) -> None:
        num_steps, rho = config.num_steps, config.rho
        sigma_min, sigma_max = config.sigma_min, config.sigma_max
        i = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas_in_log_space = sigma_max ** (1.0 / rho) + (i / (num_steps - 1.0)) * (
            sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho)
        )
        sigmas = sigmas_in_log_space ** rho
        sigmas = torch.flip(sigmas, dims=[0])
        alphas = torch.ones(num_steps, device=device, dtype=torch.float32).contiguous()
        super().__init__(alphas, sigmas)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Every scheduler type, and the class that implements it.
SCHEDULERS = {
    "flow_linear": FlowLinearScheduler,
    "ve_edm": VEEDMScheduler,
}

#: The config class each type is written against, taken from the class that implements it.
SCHEDULER_CONFIGS = {t: cls.config_cls for t, cls in SCHEDULERS.items()}


def make_scheduler(
    scheduler_config: SchedulerConfig,
    device: Optional[torch.device] = None,
) -> Scheduler:
    """Build one scheduler of the configured type, discretised into ``num_steps`` steps.

    ``scheduler_config`` is one scale's -- see ``build.scheduler_config_at_scale`` -- so
    ``num_steps`` is a plain int here.

    ``sigma_min`` is the smallest sigma the denoiser is *evaluated* at -- not the sigma of the final
    output, which is 0 (see ``Scheduler``).
    """
    if scheduler_config.type not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler type {scheduler_config.type!r}; expected one of "
                         f"{', '.join(SCHEDULERS)}")
    return SCHEDULERS[scheduler_config.type](scheduler_config, device)
