"""Optional VAE latent-space acceleration (FLUX VAE).

Only needed for a latent-space run; requires the ``diffusers`` package with the FLUX model weights.

``vae is None`` means pixel space, and the ``maybe_*`` gates pass such a run through untouched.
"""
from __future__ import annotations   # lazy annotations: `X | None` on Python 3.9, and
                                     # AutoencoderKL need not be imported at runtime

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator

import torch

from efficient_sid.timing import timed

if TYPE_CHECKING:                       # annotation-only; avoids importing diffusers for pixel runs
    from diffusers import AutoencoderKL

#: Defined here rather than in ``utils``, which imports this module.
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class LatentConfig:
    """Whether to sample in the VAE's latent space, and what dtype to load it in."""
    enabled: bool = False
    #: the VAE weights only; the dtype the rest of the run works in is ``precision.dtype``.
    vae_dtype: str = "bfloat16"


class FluxVAE:
    """The FLUX VAE, and the pixel<->latent geometry it defines.

    Wraps a diffusers ``AutoencoderKL``, reachable as ``.model`` for anything torch-level
    (``.to()``, ``.parameters()``), and applies the latent affine inside ``encode``/``decode``.

    ``encode`` returns a tensor, not the distribution ``AutoencoderKL.encode`` gives.

    Subclass and override the class attributes for a different autoencoder.
    """

    #: schnell, not dev: the same autoencoder (all 244 tensors bit-identical), but Apache-2.0 and
    #: ungated rather than gated under a non-commercial licence.
    repo = "black-forest-labs/FLUX.1-schnell"
    subfolder = "vae"
    #: Pixel dims are cropped down to a multiple of this before encoding. Must itself be a whole
    #: number of latent cells (checked against ``scale_factor``); 16 rather than 8 also keeps the
    #: latent dims even, which keeps tileable's half-shift exact.
    pixel_multiple = 16

    def __init__(self, model: AutoencoderKL) -> None:
        """Internal -- use ``from_pretrained``. ``model`` is a loaded ``AutoencoderKL``."""
        self.model = model
        #: Pixels per latent cell per dimension; 8 for the FLUX VAE.
        self.scale_factor = 2 ** (len(model.config.block_out_channels) - 1)
        if self.pixel_multiple % self.scale_factor:
            raise ValueError(
                f"pixel_multiple={self.pixel_multiple} must be a multiple of "
                f"scale_factor={self.scale_factor}, so that a cropped image is a whole number of "
                f"latent cells.")

    @classmethod
    @lru_cache(maxsize=None)      # outside @timed, so a cache hit records no "VAE load" stage
    @timed("VAE load", load=True)
    def from_pretrained(cls, device: str = 'cuda', dtype: str = 'bfloat16') -> FluxVAE:
        """Load (or reuse) the FLUX VAE onto ``device``.

        ``dtype`` is a label string (``"bfloat16"``), not a ``torch.dtype``. Results are cached per
        ``(device, dtype)``, so a long-lived process loads each combination once.
        """
        import os
        from diffusers import AutoencoderKL
        # Checked before the download starts, and by name: utils.torch_dtype would say this
        # better, but utils imports this module.
        if dtype.lower() not in _DTYPES:
            raise ValueError(
                f"Unknown latent.vae_dtype {dtype!r}; expected one of {', '.join(_DTYPES)}.")
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
        model = AutoencoderKL.from_pretrained(
            cls.repo,
            subfolder=cls.subfolder,
            torch_dtype=_DTYPES[dtype.lower()],
            device_map=device,
        )
        model.enable_tiling()
        model.enable_slicing()
        return cls(model)

    @property
    def device(self) -> torch.device:
        """Where the weights are, so where ``encode``/``decode`` run."""
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """What the weights are, so what ``encode``/``decode`` compute in."""
        return self.model.dtype

    def crop_to_multiple(self, image: torch.Tensor) -> torch.Tensor:
        """Crop a ``[C, H, W]`` pixel image, about its center, to exactly the pixels
        ``encode`` hands the VAE: H and W brought down to a multiple of ``pixel_multiple``.
        Returns ``image`` itself when they already are."""
        _, H, W = image.shape
        m = self.pixel_multiple
        h, w = (H // m) * m, (W // m) * m
        if (h, w) == (H, W):
            return image
        top, left = (H - h) // 2, (W - w) // 2
        return image[:, top:top + h, left:left + w].contiguous()

    def latent_size(self, h: int, w: int) -> tuple[int, int]:
        """Return the latent ``(H, W)`` an ``h`` x ``w`` pixel image encodes to: snapped down to
        ``pixel_multiple``, then divided by ``scale_factor``."""
        m, f = self.pixel_multiple, self.scale_factor
        return (h // m) * m // f, (w // m) * m // f

    @timed("VAE encode")
    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a ``[C, H, W]`` pixel image in [-1, 1] to a ``[C, H, W]`` latent.

        Keeps the image's dtype. The VAE runs batched, so the batch dimension is added and removed
        here. Cropped to ``pixel_multiple`` first.

        Takes the posterior mode rather than a draw from it, so one image encodes to one latent.
        """
        orig_dtype = image.dtype
        x = self.crop_to_multiple(image).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        latents = self.model.encode(x).latent_dist.mode()
        latents = (latents - self.model.config.shift_factor) * self.model.config.scaling_factor
        return latents.squeeze(0).to(orig_dtype)

    def decode(self, latents: torch.Tensor, *, tileable: bool = False) -> torch.Tensor:
        """Decode a ``[1, C, H, W]`` latent back to a ``[1, C, H, W]`` pixel image in [-1, 1],
        undoing ``encode``'s affine.

        ``latents`` is cast to the VAE's dtype first.

        ``tileable=True`` decodes through ``periodic_decoder``, so a latent whose circular
        continuation is seam-free stays seam-free in pixels.
        """
        latents = latents.to(self.dtype)
        if tileable:
            with self.periodic_decoder():
                return self._decode(latents)
        return self._decode(latents)

    # @timed sits here rather than on ``decode``, keeping the dtype cast and the padding-mode
    # swap out of the measured region.
    @timed("VAE decode", nested=True)   # nested: happens inside the sampling stage
    @torch.inference_mode()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = latents / self.model.config.scaling_factor + self.model.config.shift_factor
            return self.model.decode(latents, return_dict=False)[0]

    @contextmanager
    def periodic_decoder(self) -> Iterator[None]:
        """Temporarily make the VAE decoder periodic, so a tileable latent decodes seamlessly.

        The tileable sampler produces a latent whose circular continuation is seam-free, but the
        stock decoder zero-pads every conv and (above ~512px) decodes in tiles, both of which
        inject a border artifact exactly where the tile seam lands. Circular padding on the
        decoder convs, with tiling off, makes the decoder commute with circular shifts, so the
        decoded image is seam-free too. (The remaining decoder ops -- nearest-neighbour upsample,
        GroupNorm, global self-attention -- already commute with a circular shift.)

        Reaches into the decoder's ``Conv2d`` modules, so it is the most architecture-specific thing
        here and the first method a second autoencoder would need to override.
        """
        convs = [m for m in self.model.decoder.modules() if isinstance(m, torch.nn.Conv2d)]
        saved_modes = [m.padding_mode for m in convs]
        was_tiling = self.model.use_tiling
        try:
            for m in convs:
                m.padding_mode = "circular"
            self.model.disable_tiling()
            yield self.model
        finally:
            for m, mode in zip(convs, saved_modes):
                m.padding_mode = mode
            if was_tiling:
                self.model.enable_tiling()


# ---------------------------------------------------------------------------
# The gates: a pixel-space run (vae is None) passes straight through.
# ---------------------------------------------------------------------------

def maybe_load_vae(latent_config: LatentConfig, device: str | torch.device) -> FluxVAE | None:
    """Load the FLUX VAE when the config enables latent space, else return ``None``.

    Loaded once (lru_cached) and timed.
    """
    if not latent_config.enabled:
        return None
    return FluxVAE.from_pretrained(str(device), latent_config.vae_dtype)


def maybe_encode(image: torch.Tensor, vae: FluxVAE | None) -> torch.Tensor:
    """Encode ``image`` to ``vae``'s latent space; pass it through when ``vae`` is None."""
    if vae is None:
        return image
    return vae.encode(image)


def maybe_decode(x: torch.Tensor, vae: FluxVAE | None, *, tileable: bool = False) -> torch.Tensor:
    """Decode a latent back to pixels; pass ``x`` through when ``vae`` is None.

    ``x`` is ``[1, C, H, W]``, the shape the samplers hold per scale.

    ``tileable=True`` decodes through a periodic decoder (see ``FluxVAE.periodic_decoder``); it is
    keyword-only because it changes which decoder runs.
    """
    if vae is None:
        return x
    return vae.decode(x, tileable=tileable)
