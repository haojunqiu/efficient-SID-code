"""What each application can be configured with, and how a config is loaded.

One class per application, naming its inputs, its outputs, and the groups it configures. Each
group's class lives with the code that reads it -- ``PyramidConfig`` in ``pyramid.py``,
``ClosedFormDenoiserConfig`` in ``closed_form_denoiser.py``, ``TilingConfig`` in
``applications/tileable.py``.

``parse_args`` is what the scripts call: it reads ``--config`` and ``key=value`` overrides, then
``load`` merges them onto the class in struct mode.

Nothing else in the package imports this module, so the sampling code runs without a config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from omegaconf import MISSING, OmegaConf

from efficient_sid.applications.symmetric import SymmetryConfig
from efficient_sid.applications.text_style import ClipGuidanceConfig
from efficient_sid.applications.tileable import TilingConfig
from efficient_sid.build import AutoConfig
from efficient_sid.closed_form_denoiser import CLOSED_FORM_DENOISER_CONFIGS
from efficient_sid.clip_extractor import ClipExtractorConfig
from efficient_sid.image_denoiser import ImageDenoiserConfig
from efficient_sid.latent import LatentConfig
from efficient_sid.pyramid import PyramidConfig
from efficient_sid.scheduler import SCHEDULER_CONFIGS, FlowLinearConfig, SchedulerConfig
from efficient_sid.utils import PrecisionConfig


@dataclass
class DiagnosticsConfig:
    """Opt-in extra output while a run proceeds. Neither field changes the result."""
    #: print the wall-time breakdown at the end.
    report_timing: bool = True
    #: set a path to dump intermediate results: each pyramid scale's finished result, or
    #: each step's for the single-scale text_style.
    intermediate_output_dir: Optional[str] = None


@dataclass
class AppConfig:
    """What all six applications read. Each declares the rest for itself."""
    seed: int = 10
    #: how many samples to draw. The pyramid, denoisers and index are built once and reused;
    #: sample i uses ``seed + i`` and matches a single run at that seed.
    num_samples: int = 1
    final_output_path: Optional[str] = None
    #: also write <name>_preview.jpg at this long side; null = off.
    preview_long_side: Optional[int] = None
    image_denoiser: ImageDenoiserConfig = field(default_factory=ImageDenoiserConfig)
    scheduler: SchedulerConfig = field(default_factory=FlowLinearConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass
class UncondAppConfig(AppConfig):
    image_path: str = MISSING
    input_resize_to: Optional[str] = None
    #: "(H, W)" in pixels for the grid to sample on, read by ``build.resolve_sample_size``.
    #: Unset, the grid is the input's own size. A string or a list of two ints, hence untyped.
    output_size: Any = None
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)


@dataclass
class RetargetAppConfig(AppConfig):
    """Samples the input onto a different grid, given by ``output_size``."""
    image_path: str = MISSING
    input_resize_to: Optional[str] = None
    #: "(H, W)" in pixels for the grid to sample on, read by ``build.resolve_sample_size``.
    #: Unset, the grid is the input's own size. A string or a list of two ints, hence untyped.
    output_size: Any = None
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)


@dataclass
class SymmetricAppConfig(AppConfig):
    image_path: str = MISSING
    input_resize_to: Optional[str] = None
    #: "(H, W)" in pixels for the grid to sample on, read by ``build.resolve_sample_size``.
    #: Unset, the grid is the input's own size. A string or a list of two ints, hence untyped.
    output_size: Any = None
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)
    symmetry: SymmetryConfig = field(default_factory=SymmetryConfig)


@dataclass
class TileableAppConfig(AppConfig):
    image_path: str = MISSING
    input_resize_to: Optional[str] = None
    #: "(H, W)" in pixels for the grid to sample on, read by ``build.resolve_sample_size``.
    #: Unset, the grid is the input's own size. A string or a list of two ints, hence untyped.
    output_size: Any = None
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)


@dataclass
class StructuralAnalogyAppConfig(AppConfig):
    """Samples the structure image's layout with the style image's patches.

    Both are resized so their short side is ``output_short_side``, which makes a patch cover the
    same fraction of each.
    """
    style_image_path: str = MISSING
    structure_image_path: str = MISSING
    output_short_side: Optional[int] = None
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)
    noise_level: Optional[float] = None


@dataclass
class TextStyleAppConfig(AppConfig):
    """Restyles one image toward ``text``, at the image's own resolution.

    Single-scale: inversion already carries the content image's full-resolution layout. Pixel-space
    only: CLIP guidance would have to backpropagate through the VAE decoder.
    """
    image_path: str = MISSING
    input_resize_to: Optional[str] = None
    text: str = MISSING
    noise_level: float = 0.3
    clip_extractor: ClipExtractorConfig = field(default_factory=ClipExtractorConfig)
    clip_guidance: ClipGuidanceConfig = field(default_factory=ClipGuidanceConfig)


def _with_type_class(declared: Any, written: Any, path: str, by_type: Dict[str, type]) -> Any:
    """Return ``declared`` with the block at ``path`` replaced by the class its ``type`` names.

    Each type has its own config class, so which class to validate against is not known until
    ``type`` has been read -- hence this runs between reading the written values and merging them.

    ``"auto"``, and one entry per pyramid scale, name no single backend: the choice is not made
    until ``resolve`` sees the image. Such a block is left unvalidated: with no single class to
    check against, its keys are accepted as written.
    """
    node = OmegaConf.select(declared, path, default=None)
    if node is None:
        return declared
    selected = OmegaConf.select(written, f"{path}.type", default=None)
    if selected is None:
        return declared
    if not isinstance(selected, str) or selected == "auto":
        OmegaConf.set_struct(node, False)
        return declared
    cls = by_type.get(selected)
    if cls is None:
        raise ValueError(f"Unknown {path} type {selected!r}; expected one of "
                         f"{', '.join(by_type)}")
    OmegaConf.update(declared, path, OmegaConf.structured(cls), force_add=True)
    return declared


def load(app_config_cls: type, config_path: Union[str, Path], overrides: Sequence[str] = ()) -> Any:
    """Build one application's config from its class, a YAML file and ``key=value`` overrides.

    The config is built from ``app_config_cls``, so a key the class does not declare raises, a
    value of the wrong type raises, and a field left ``???`` raises when read.
    """
    declared = OmegaConf.structured(app_config_cls)
    file_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_dotlist(list(overrides))

    written = OmegaConf.merge(file_config, cli_config)
    declared = _with_type_class(declared, written, "scheduler", SCHEDULER_CONFIGS)
    declared = _with_type_class(
        declared,
        written,
        "image_denoiser.patch_denoiser",
        CLOSED_FORM_DENOISER_CONFIGS,
    )
    return OmegaConf.merge(declared, file_config, cli_config)


def parse_args(app_config_cls: type) -> Any:
    """Build this application's config from ``--config`` and ``key=value`` overrides."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Additional overrides in key=value format",
    )
    args = parser.parse_args()
    return load(app_config_cls, args.config, args.overrides)
