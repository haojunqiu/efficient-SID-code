"""Multi-scale pyramid processing with Laplacian consistency.

``BasePyramidProcessor`` is the algorithm -- pyramid geometry and the two-scale Laplacian blend
-- over ``_downsample`` / ``_upsample`` that a subclass supplies. ``PyramidProcessor`` is the one
concrete subclass, resampling with the vendored ``resize_right``.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from efficient_sid.resize_right import resize
from efficient_sid.timing import BLEND_TIME  # hot-path GPUTime for Laplacian blending; lives in timing


@dataclass
class PyramidConfig:
    """``PyramidProcessor``'s arguments."""
    #: the size ratio between adjacent scales; 0.5 halves each side going coarser.
    relative_scale: float = 0.5
    #: an int, or "auto" to derive from the image's short side. Untyped because OmegaConf cannot
    #: express `Union[int, str]`; ``build.resolve`` derives the "auto" case.
    num_scales: Any = 4
    #: the coarsest scale's short side. With num_scales "auto" the depth is chosen so the coarsest
    #: lands exactly here; with an explicit depth it floors the coarsest instead.
    #: 1.5-2x patch_size is the usable range: below it samples follow the input more closely and
    #: vary less between seeds, above it they keep less of its global structure. null uses
    #: 1.7x, one pick from that range.
    min_short_side: Optional[int] = None


class BasePyramidProcessor:
    """Gaussian pyramid geometry plus the two-scale Laplacian blend, over ``_downsample`` /
    ``_upsample`` supplied by a subclass.

    Shape convention: every method here expects ``[..., H, W]`` shape, where ``...`` means an
    arbitrary number of leading dimensions -- only the trailing two are resampled, and the
    leading ones are passed through. Shapes are always *full* shapes, not just spatial extents:
    ``gaussian_pyramid_shapes`` returns and ``_downsample`` / ``_upsample`` accept tuples that
    still carry the leading dimensions.
    """

    def __init__(self, config: PyramidConfig) -> None:
        self.relative_scale = config.relative_scale
        self.num_scales = config.num_scales
        self.min_short_side = config.min_short_side

    def gaussian_pyramid_shapes(
        self,
        base_shape: Sequence[int],
        num_scales: Optional[int] = None,
        min_short_side: Optional[int] = None,
    ) -> List[Tuple[int, ...]]:
        """Return the shape of every pyramid scale, finest (the base shape) first, coarsest last.

        ``base_shape`` is a *shape*, not a tensor -- any sequence of length >= 2 whose last two
        entries are (H, W). The returned tuples repeat ``base_shape``'s leading dims verbatim and
        shrink only the trailing two, so they can be handed straight to ``_downsample``.

        ``num_scales`` counts the scales, so ``len(result) == num_scales``.
        """
        num_scales = num_scales if num_scales is not None else self.num_scales
        assert num_scales is not None
        min_short_side = min_short_side if min_short_side is not None else self.min_short_side

        base_shape = tuple(base_shape)
        prefix = base_shape[:-2]
        H0, W0 = base_shape[-2:]

        shapes = [base_shape]
        for l in range(1, num_scales):
            H_l = math.ceil(H0 * (self.relative_scale ** l))
            W_l = math.ceil(W0 * (self.relative_scale ** l))
            shapes.append((*prefix, H_l, W_l))

        if min_short_side is not None and num_scales > 1:
            # min_short_side rescales only the coarsest scale, clamped to the scale above it. If
            # that scale is already below min_short_side the clamp swallows the bump and
            # min_short_side silently does nothing -- so refuse rather than return scales that
            # ignore it.
            prev_H, prev_W = shapes[-2][-2:]
            if min(prev_H, prev_W) < min_short_side:
                short0 = min(H0, W0)
                max_ns = (2 + int(math.floor(math.log2(short0 / min_short_side)))
                          if short0 >= min_short_side else 1)
                raise ValueError(
                    f"num_scales={num_scales} is too large for min_short_side={min_short_side}: the "
                    f"second-coarsest scale is {prev_H}x{prev_W} (short side {min(prev_H, prev_W)} "
                    f"< {min_short_side}), so min_short_side would be silently clamped to a no-op. "
                    f"Use num_scales <= {max_ns}.")

            *_, H_last, W_last = shapes[-1]
            short_last = min(H_last, W_last)
            if short_last < min_short_side:
                short0 = min(H0, W0)
                # exact ceil division
                H_adj = min(-(-H0 * min_short_side // short0), prev_H)
                W_adj = min(-(-W0 * min_short_side // short0), prev_W)
                shapes[-1] = (*prefix, H_adj, W_adj)

        return shapes

    # --- subclasses must implement these ---
    def _downsample(self, image: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        """Resample ``image`` down to ``output_shape`` (a full shape; only its last two dims differ
        from ``image.shape``). Returns a tensor of exactly ``output_shape``."""
        raise NotImplementedError

    def _upsample(self, image: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        """Resample ``image`` up to ``output_shape``, same shape convention as ``_downsample``.
        Returns a tensor of exactly ``output_shape``."""
        raise NotImplementedError

    # --- common logic (uses _downsample / _upsample) ---
    def build_gaussian_pyramid(
        self,
        image: torch.Tensor,
        num_scales: Optional[int] = None,
        min_short_side: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Build a Gaussian pyramid of ``image``, finest (``image`` itself) first, coarsest last.

        ``image`` has ``[..., H, W]`` shape, so ``(C, H, W)`` and ``(1, C, H, W)`` both work and
        every scale keeps the input's leading dimensions. Each scale is downsampled from the one
        above it, not from the base, so successive scales stay consistent with each other.
        """
        num_scales = num_scales if num_scales is not None else self.num_scales
        shapes = self.gaussian_pyramid_shapes(
            image.shape,
            num_scales=num_scales,
            min_short_side=min_short_side,
        )
        pyramid = [image]
        for s in shapes[1:]:
            pyramid.append(self._downsample(pyramid[-1], s))
        return pyramid

    def blend_two_scale(self, fine: torch.Tensor, coarser: torch.Tensor) -> torch.Tensor:
        """Enforce Laplacian consistency between two adjacent scales.

        Keeps ``fine``'s detail band and takes its low-frequency content from ``coarser``, the
        already-finished scale above it, which is what stops the finer scale from drifting away
        from the layout the coarse scale settled::

            low    = down(fine)          # at gaussian_pyramid_shapes(fine.shape, 2)[1]
            detail = fine - up(low)      # fine's Laplacian band
            out    = up(coarser) + detail

        ``fine`` is ``(1, C, H, W)`` and ``coarser`` is ``(1, C, h, w)`` with h, w smaller; the
        two need no particular size relationship, since each is resampled to ``fine``'s spatial
        size on its own. Returns ``(1, C, H, W)``.
        """
        dtype = fine.dtype
        # Resample as 3-D and cast back at the end: ``resize`` is not bit-identical for 4-D
        # input, and it builds float32 weights whatever dtype it is handed.
        fine, coarser = fine.squeeze(0), coarser.squeeze(0)
        with BLEND_TIME.record():
            low_shape = self.gaussian_pyramid_shapes(fine.shape, num_scales=2)[1]
            detail = fine - self._upsample(self._downsample(fine, low_shape), fine.shape)
            blended = self._upsample(coarser, fine.shape) + detail
        return blended.unsqueeze(0).to(dtype)


class PyramidProcessor(BasePyramidProcessor):
    """Pyramid processor using the vendored resize_right library.

    Follows the shape convention documented on ``BasePyramidProcessor``.
    """

    def _downsample(self, image: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        return resize(image, scale_factor=None, output_shape=output_shape)

    def _upsample(self, image: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        return resize(image, scale_factor=None, output_shape=output_shape)
