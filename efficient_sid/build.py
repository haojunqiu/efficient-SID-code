"""Assemble the standard pipeline from an application's config.

The six CLI scripts all do the same four things -- resolve the config for this run, build one
scheduler per scale, build the Gaussian pyramid, construct one image denoiser per scale -- so those
live here.

A config may leave its size-driven fields as ``"auto"`` rather than naming numbers, in which case
``resolve`` works them out from the image before any of the above runs.
"""

import ast
import copy
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
from omegaconf import ListConfig, OmegaConf

from efficient_sid.closed_form_denoiser import (CLOSED_FORM_DENOISER_CONFIGS,
                                               make_closed_form_denoiser)
from efficient_sid.image_denoiser import ImageDenoiserConfig, ImageDenoiser, extract_patches
from efficient_sid.pyramid import PyramidConfig, PyramidProcessor
from efficient_sid.latent import FluxVAE
from efficient_sid.scheduler import Scheduler, SchedulerConfig, make_scheduler


# -------------------------------------------------------------------------
# Making a config concrete
#
# ``resolve`` derives the "auto" fields from two sizes -- the one patches come from, and the grid
# being sampled on -- and returns the config for this run. A script calls it once, before it
# builds anything.
# -------------------------------------------------------------------------

def _coerce_scalar(x: Any) -> Any:
    """Convert a string to the Python value it represents; pass anything else through."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def parse_size(value: Any, field: str = "") -> Tuple[int, int]:
    """Read a config size into an ``(H, W)`` pair of ints.

    Sizes are written as ``"(H, W)"`` strings or as two-item lists; both YAML and CLI overrides
    use the string form, since OmegaConf parses only its own list syntax.

    ``field`` names the key being read, so a bad value says which one to fix.
    """
    size = _coerce_scalar(value)
    if not isinstance(size, str):
        try:
            h, w = size
            return int(h), int(w)
        except (TypeError, ValueError):
            pass
    raise ValueError(
        f"Expected a size as (H, W), got {field + '=' if field else ''}{value!r}")


def resolve_sample_size(
    config: Any,
    ref_size: Sequence[int],
    vae: Optional[FluxVAE],
) -> Tuple[int, int]:
    """Resolve the height and width to sample at.

    Return the reference's size unchanged if no ``output_size`` is configured. Otherwise parse the
    configured size and, when a VAE is given, convert it to latent dimensions, snapping to the
    VAE's ``pixel_multiple`` first.
    """
    output_size = config.get("output_size")
    if output_size is None:
        return tuple(ref_size)
    h, w = parse_size(output_size, "output_size")
    if vae is None:
        print(f"[sample size] {h}x{w} px")
        return (h, w)
    lh, lw = vae.latent_size(h, w)
    print(f"[sample size] {h}x{w} px requested -> {lh * vae.scale_factor}x{lw * vae.scale_factor} "
          f"px -> latent {lh}x{lw}")
    return (lh, lw)


@dataclass
class AutoConfig:
    """The sizes at which ``resolve`` switches an ``"auto"`` field to its next setting.

    A scale uses ann once its dataset passes ``denoiser_type_patch_threshold`` patches, is strided
    2x once its query grid passes ``query_stride_patch_threshold`` positions, and is chunked to
    ~``query_grid_per_chunk`` positions per chunk, counted before stride.

    Tuned so a gigapixel run fits on a 96 GB GPU at patch_size 7 in latent bfloat16. On other
    hardware, override these or set the fields they derive explicitly.
    """
    denoiser_type_patch_threshold: int = 250_000
    query_stride_patch_threshold: int = 8_000_000
    query_grid_per_chunk: int = 7_500


AUTO_FIELDS = ("pyramid.num_scales", "image_denoiser.patch_denoiser.type",
               "image_denoiser.query_stride", "image_denoiser.patch_denoiser.query_chunks")


def reject_auto(config: Any, auto_fields: Sequence[str] = ()) -> None:
    """Raise if ``config`` leaves a field as ``"auto"`` that this application does not derive.

    ``resolve`` calls this for its callers. An application outside the per-scale machinery calls it
    directly -- text_style is single-scale and reads these fields straight off the config, where an
    unresolved ``"auto"`` would otherwise surface as an error from the denoiser it reached.
    """
    refused = [f for f in AUTO_FIELDS
               if f not in auto_fields and OmegaConf.select(config, f) == "auto"]
    if refused:
        raise ValueError(
            f"This application does not derive {', '.join(refused)}; set "
            f"{'them' if len(refused) > 1 else 'it'} explicitly.")


# --- The rules themselves: plain numbers in, plain values out.

def auto_num_scales(short_side: int, min_short_side: int) -> int:
    """Return how many scales it takes for the coarsest to reach ``min_short_side``, halving each
    step.

    The count rounds up, so halving lands the coarsest at or below ``min_short_side`` and
    ``PyramidProcessor``'s floor brings it exactly there.
    """
    if min_short_side > short_side:
        raise ValueError(
            f"min_short_side={min_short_side} exceeds the {short_side} px short side it is measured "
            f"against, so no pyramid reaches it. Lower min_short_side, or check the sizes passed in.")
    n = 1
    while min_short_side * 2 ** (n - 1) < short_side:
        n += 1
    return n


def auto_denoiser_type(dict_counts: Sequence[int], threshold: int) -> List[str]:
    """Pick ``ann`` on scales whose patch dataset exceeds ``threshold``, ``exact_flash_attn`` below
    (where exact is both faster and exact). One entry per scale."""
    return ["ann" if c > threshold else "exact_flash_attn" for c in dict_counts]


def auto_query_stride(query_counts: Sequence[int], threshold: int) -> List[int]:
    """Use stride 2 on scales whose query grid exceeds ``threshold``, else 1. One entry per scale."""
    return [2 if c > threshold else 1 for c in query_counts]


def auto_query_chunks(query_counts: Sequence[int], per_chunk: int) -> List[int]:
    """Split each scale's query grid into chunks of ~``per_chunk`` grid positions, at least 1.

    ``query_counts`` is the full grid *before* stride, so this equals the real patches per chunk
    only where stride is 1; a strided scale runs ~``per_chunk / stride**2`` real queries per chunk.
    """
    return [max(1, round(c / per_chunk)) for c in query_counts]


def _resolve_auto_fields(
    config: Any,
    ref_size: Tuple[int, ...],
    sample_size: Tuple[int, ...],
    auto_fields: Sequence[str],
) -> None:
    """Fill in the fields set to ``"auto"``, in place, each from the size that governs it.

    Which size governs which field:

    - ``num_scales`` from the smaller short side of ``ref_size`` and ``sample_size`` -- patches are
      extracted at one and placed into the other, so both have to stay large enough to hold a patch.
    - ``denoiser_type`` from ``ref_size``, where the patch dataset comes from.
    - ``query_stride`` and ``query_chunks`` from ``sample_size``, where the query patches live.

    ``denoiser_type``, ``query_stride`` and ``query_chunks`` are decided by thresholds in
    ``AutoConfig``, which are tuned to the GPU. ``num_scales`` uses no threshold: it follows from
    the short side and ``min_short_side``.

    Each field resolved from ``"auto"`` prints what it derived and which knob (or default) drove it;
    a knob set while its field is *not* ``"auto"`` is ignored and warns.
    """
    reject_auto(config, auto_fields)

    image_denoiser_config = config.image_denoiser
    patch_denoiser_config = image_denoiser_config.patch_denoiser
    patch_size = image_denoiser_config.patch_size

    # Each auto field is governed by one knob in the ``auto`` group. ``min_short_side`` is not
    # in this table: it also floors the coarsest pyramid scale, so it is used whether or not
    # num_scales is auto and must not warn as "ignored".
    KNOBS = {"image_denoiser.patch_denoiser.type": "denoiser_type_patch_threshold",
             "image_denoiser.query_stride": "query_stride_patch_threshold",
             "image_denoiser.patch_denoiser.query_chunks": "query_grid_per_chunk"}
    defaults = AutoConfig()
    # A config loaded through its application class always has this group; one loaded straight from
    # YAML (a harness, a notebook) need not, and then every knob is at its default.
    auto_config = OmegaConf.select(config, "auto") or defaults

    def is_set(knob: str) -> bool:
        """Return whether this run gave the knob a value of its own."""
        return getattr(auto_config, knob) != getattr(defaults, knob)

    for field, knob in KNOBS.items():
        if OmegaConf.select(config, field) != "auto" and is_set(knob):
            warnings.warn(f"'{knob}' is set but '{field}' is not 'auto', so '{knob}' is ignored.")

    def knob_of(field: str) -> Tuple[Any, str]:
        knob = KNOBS[field]
        value = getattr(auto_config, knob)
        return value, f"{knob}={value}{'' if is_set(knob) else ' (default)'}"

    def note(field: str, resolved: Any, reason: str) -> None:
        print(f"[resolve] {field}=auto -> {resolved}  (via {reason})")

    if config.pyramid.num_scales == "auto":
        short_side = min(*ref_size, *sample_size)
        configured_min_short_side = config.pyramid.min_short_side
        # 1.7x is one pick from the 1.5-2x range in PyramidConfig.min_short_side, not a
        # measured optimum; other values in that range work as defaults too.
        min_short_side = configured_min_short_side or round(1.7 * patch_size)
        config.pyramid.num_scales = auto_num_scales(short_side, min_short_side)
        note(
            "pyramid.num_scales",
            config.pyramid.num_scales,
            f"short side {short_side}, "
            f"min_short_side={min_short_side}"
            f"{'' if configured_min_short_side else ' (default 1.7*patch_size)'}",
        )
    pyramid_processor = PyramidProcessor(config.pyramid)

    def patch_counts(size: Sequence[int]) -> List[int]:
        return [max(0, h - patch_size + 1) * max(0, w - patch_size + 1)
                for (h, w) in pyramid_processor.gaussian_pyramid_shapes(size)]

    if patch_denoiser_config.get("type") == "auto":
        threshold, reason = knob_of("image_denoiser.patch_denoiser.type")
        patch_denoiser_config.type = auto_denoiser_type(patch_counts(ref_size), threshold)
        note("image_denoiser.patch_denoiser.type", list(patch_denoiser_config.type), reason)

    if (image_denoiser_config.get("query_stride") == "auto"
            or patch_denoiser_config.get("query_chunks") == "auto"):
        # Query patches come from the sample grid, not the reference dataset.
        query_counts = patch_counts(sample_size)
        if image_denoiser_config.get("query_stride") == "auto":
            cutoff, reason = knob_of("image_denoiser.query_stride")
            image_denoiser_config.query_stride = auto_query_stride(query_counts, cutoff)
            note("image_denoiser.query_stride", list(image_denoiser_config.query_stride), reason)
        if patch_denoiser_config.get("query_chunks") == "auto":
            per_chunk, reason = knob_of("image_denoiser.patch_denoiser.query_chunks")
            patch_denoiser_config.query_chunks = auto_query_chunks(query_counts, per_chunk)
            note(
                "image_denoiser.patch_denoiser.query_chunks",
                list(patch_denoiser_config.query_chunks),
                reason,
            )


def resolve_per_scale(val: Any, num_scales: int, field: str = "") -> List[Any]:
    """Expand one config value into a list with an entry per scale.

    A scalar broadcasts to every scale; a list (or a string holding a list literal, which is how
    a CLI override arrives) must already be the right length. Strings are coerced to the value
    they spell, so `num_steps="[10,10,50]"` from the command line behaves like the YAML list.

    ``field`` names the key being expanded, so a wrong-length list says which one to fix.
    """
    # Already a container
    if isinstance(val, (list, tuple, ListConfig)):
        out = [_coerce_scalar(v) for v in list(val)]
        if len(out) != num_scales:
            raise ValueError(
                f"{field or 'This value'} takes one entry per pyramid scale: "
                f"expected {num_scales}, got {len(out)}.")
        return out
    # String that might be a list literal or scalar
    if isinstance(val, str):
        s = val.strip()
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            out = [_coerce_scalar(v) for v in parsed]
            if len(out) != num_scales:
                raise ValueError(
                f"{field or 'This value'} takes one entry per pyramid scale: "
                f"expected {num_scales}, got {len(out)}.")
            return out
        return [_coerce_scalar(val)] * num_scales
    # Non-string scalar
    return [val] * num_scales


def _resolve_per_scale_fields(config: Any) -> None:
    """Give every per-scale field one entry per scale, in place.

    A scalar broadcasts to every scale (the common case); a list sets each scale independently.
    """
    ns = config.pyramid.num_scales
    image_denoiser_config = config.image_denoiser
    patch_denoiser_config = image_denoiser_config.patch_denoiser
    image_denoiser_config.dataset_stride = resolve_per_scale(
        image_denoiser_config.get("dataset_stride", 1),
        ns,
        "image_denoiser.dataset_stride",
    )
    image_denoiser_config.query_stride = resolve_per_scale(
        image_denoiser_config.query_stride, ns, "image_denoiser.query_stride")
    image_denoiser_config.fold_rho = resolve_per_scale(
        image_denoiser_config.fold_rho, ns, "image_denoiser.fold_rho")
    patch_denoiser_config.query_chunks = resolve_per_scale(
        patch_denoiser_config.get("query_chunks", 1),
        ns,
        "image_denoiser.patch_denoiser.query_chunks",
    )
    patch_denoiser_config.type = resolve_per_scale(
        patch_denoiser_config.get("type", "exact"),
        ns,
        "image_denoiser.patch_denoiser.type",
    )
    config.scheduler.num_steps = resolve_per_scale(
        config.scheduler.num_steps, ns, "scheduler.num_steps")


def resolve(
    config: Any,
    ref_size: Sequence[int],
    sample_size: Sequence[int],
    auto_fields: Sequence[str] = (),
) -> Any:
    """Return a copy of ``config`` with every ``"auto"`` and per-scale field filled in.

    The config passed in is not modified. (Not ``OmegaConf.resolve``, which evaluates ``${...}``
    interpolations in place.)

    ``ref_size`` is the size of the image the patch dataset comes from; ``sample_size`` is the
    grid being sampled on -- both ``(H, W)``, in latent cells under ``latent.enabled`` and pixels
    otherwise.

    ``auto_fields`` lists the fields this application derives; leaving any other field ``"auto"``
    raises.
    """
    resolved = copy.deepcopy(config)
    _resolve_auto_fields(resolved, tuple(ref_size), tuple(sample_size), auto_fields)
    _resolve_per_scale_fields(resolved)
    return resolved


# -------------------------------------------------------------------------
# Building the objects: schedulers, pyramid, denoisers
# -------------------------------------------------------------------------

def scheduler_config_at_scale(scheduler_config: SchedulerConfig, scale: int) -> SchedulerConfig:
    """Return this scale's scheduler config, with ``num_steps`` a plain int.

    Expects a config ``resolve`` has expanded, so ``num_steps`` holds one entry per scale.
    Everything else -- the type, ``sigma_min``, ``eta``, and whatever settings that type adds --
    is the same at every scale, so the returned config keeps the class it came in as.
    """
    one_scale = copy.deepcopy(scheduler_config)
    one_scale.num_steps = scheduler_config.num_steps[scale]
    return one_scale


def build_schedulers(
    scheduler_config: SchedulerConfig,
    num_scales: int,
    device: torch.device,
) -> List[Scheduler]:
    """Build one ``Scheduler`` per scale, coarsest last.

    Only ``num_steps`` can differ between scales; ``resolve`` has already expanded it to one entry
    per scale, usually the same value repeated. Every other setting is shared -- making another
    one per-scale means accepting a list for it here and in the config.
    """
    return [make_scheduler(scheduler_config_at_scale(scheduler_config, s), device)
            for s in range(num_scales)]


def build_pyramid(
    pyramid_config: PyramidConfig,
    image: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[PyramidProcessor, List[torch.Tensor]]:
    """Build a Gaussian pyramid of ``image`` at the config's scales. Returns ``(processor, pyramid)``
    ``pyramid`` is a list of (C, H, W) tensors, coarsest last, cast to ``dtype``."""
    processor = PyramidProcessor(pyramid_config)
    return processor, [img.to(dtype) for img in processor.build_gaussian_pyramid(image)]


def image_denoiser_config_at_scale(
    image_denoiser_config: ImageDenoiserConfig,
    scale: int,
) -> ImageDenoiserConfig:
    """Return the denoiser config for one pyramid scale, every per-scale field a plain scalar.

    Expects a config ``resolve`` has expanded, so ``dataset_stride``, ``query_stride``,
    ``fold_rho``, ``type`` and ``query_chunks`` each hold one entry per scale. Everything else --
    ``patch_size``, ``bottleneck_dtype``, and the ann settings -- is the same at every scale.

    The returned ``patch_denoiser`` is the class this scale's backend needs, which is narrower than
    the config was authored against whenever that named ``"auto"``.
    """
    all_scales = image_denoiser_config.patch_denoiser
    scale_type = all_scales.type[scale]

    # resolve has already turned any "auto" into a concrete backend, so this scale names one and
    # gets that backend's class -- narrower than the one the config was authored against.
    patch_denoiser_cls = CLOSED_FORM_DENOISER_CONFIGS[scale_type]
    patch_denoiser = OmegaConf.structured(patch_denoiser_cls)
    for name in patch_denoiser_cls.__dataclass_fields__:
        # A block whose type was not committed at load carries only what the file wrote, so a
        # setting it left out keeps this class's default.
        if name in all_scales:
            setattr(patch_denoiser, name, getattr(all_scales, name))
    patch_denoiser.type = scale_type
    patch_denoiser.query_chunks = all_scales.query_chunks[scale]

    one_scale = OmegaConf.structured(ImageDenoiserConfig)
    one_scale.patch_size = image_denoiser_config.patch_size
    one_scale.dataset_stride = image_denoiser_config.dataset_stride[scale]
    one_scale.query_stride = image_denoiser_config.query_stride[scale]
    one_scale.fold_rho = image_denoiser_config.fold_rho[scale]
    one_scale.patch_denoiser = patch_denoiser
    return one_scale


def build_image_denoiser(
    image_denoiser_config: ImageDenoiserConfig,
    img: torch.Tensor,
    dtype: torch.dtype,
) -> ImageDenoiser:
    """Build one scale's ``ImageDenoiser``: extract the image's patches into a closed-form
    denoiser of the chosen backend, wrapped for image-level (extract -> denoise -> fold)
    operation, on CUDA.

    ``image_denoiser_config`` is one scale's -- see ``image_denoiser_config_at_scale`` -- so its
    per-scale fields are scalars here. The noise level reaches the denoiser per call, through
    ``ImageDenoiser.forward``, so no scheduler is needed to build one."""
    return ImageDenoiser(
        config=image_denoiser_config,
        patch_denoiser=make_closed_form_denoiser(
            image_denoiser_config.patch_denoiser,
            dataset=extract_patches(
                image=img,
                patch_size=image_denoiser_config.patch_size,
                stride=image_denoiser_config.dataset_stride,
                mask=None,
            ),
        ),
        dtype=dtype,
    ).cuda()


def build_image_denoisers(
    image_denoiser_config: ImageDenoiserConfig,
    pyramid: Sequence[torch.Tensor],
    skip_scales: Sequence[int] = (),
    dtype: torch.dtype = torch.float32,
) -> List[Optional[ImageDenoiser]]:
    """Build one ``ImageDenoiser`` per pyramid scale, coarsest last -- the list every sampler takes.

    ``skip_scales`` holds the scale indices the application never denoises, which are left as
    ``None`` rather than omitted so the list stays indexable by scale: retarget takes its coarsest
    scale straight from the resized image.

    A denoised scale must be at least ``patch_size`` on both sides, or it extracts no patches and
    raises. Skipped scales are exempt, so a pyramid may end on a scale too small to denoise.
    """
    denoisers = []
    for s, img in enumerate(pyramid):
        if s in skip_scales:
            denoisers.append(None)
            continue
        *_, H, W = img.shape
        if min(H, W) < image_denoiser_config.patch_size:
            raise ValueError(
                f"Pyramid scale {s} is {H}x{W}, smaller than patch_size={image_denoiser_config.patch_size}, "
                f"so it extracts zero patches. Reduce num_scales, or set min_short_side >= patch_size.")
        denoisers.append(build_image_denoiser(
            image_denoiser_config_at_scale(image_denoiser_config, s),
            img,
            dtype,
        ))
    return denoisers


# -------------------------------------------------------------------------
# Reporting a run's setup
#
# What the config resolved to, and what was built from it, printed before sampling starts.
# ``StageTimer.report`` prints the wall-time counterpart afterwards.
# -------------------------------------------------------------------------

#: Floor for both this module's banner and ``StageTimer.report``'s, so a run's two blocks match.
BANNER_WIDTH = 66


def report_setup(title: str, lines: Sequence[str] = ()) -> None:
    """Print a run's derived setup, banner-matched to ``StageTimer.report``."""
    width = max(BANNER_WIDTH, max((len(line) for line in lines), default=0) + 2)
    print(f"\n{'=' * width}")
    print(f"{title} — setup")
    print(f"{'-' * width}")
    for line in lines:
        print(f"  {line}")
    print(f"{'=' * width}")


def report_scale_table(
    title: str,
    config: Any,
    denoisers: Sequence[Optional[ImageDenoiser]],
    pyramid: Sequence[torch.Tensor],
    sample_pyramid_shapes: Sequence[Sequence[int]],
    vae: Optional[FluxVAE] = None,
    lines: Sequence[str] = (),
) -> None:
    """Print the setup with one row per pyramid scale: the dataset each denoiser holds, and
    the grid it samples onto.

    A column whose value is the same at every scale is listed once above the table instead.
    ``lines`` are the application's own facts, printed first.
    """
    patch_size = config.image_denoiser.patch_size
    columns = [
        ("Dataset (HxW)", [f"{scale.shape[-2]}x{scale.shape[-1]}" for scale in pyramid]),
        ("Output (HxW)", [f"{shape[-2]}x{shape[-1]}" for shape in sample_pyramid_shapes]),
        ("#Patches", [d.patch_denoiser.N if d is not None else "-" for d in denoisers]),
        ("dataset_stride", config.image_denoiser.dataset_stride),
        ("Short/Patch", [f"{min(scale.shape[-2:]) / patch_size:.1f}" for scale in pyramid]),
        ("denoiser_type", config.image_denoiser.patch_denoiser.type),
        ("num_steps", config.scheduler.num_steps),
        ("query_stride", config.image_denoiser.query_stride),
        ("fold_rho", config.image_denoiser.fold_rho),
        ("query_chunks", config.image_denoiser.patch_denoiser.query_chunks),
    ]
    columns = [(name, [str(v) for v in values]) for name, values in columns]
    # The output grid repeats the dataset unless the application samples onto a different one.
    if dict(columns)["Output (HxW)"] == dict(columns)["Dataset (HxW)"]:
        columns = [c for c in columns if c[0] != "Output (HxW)"]

    varying = [(name, values) for name, values in columns if len(set(values)) > 1]
    order = ["denoiser_type", "num_steps", "dataset_stride", "query_stride", "fold_rho",
             "query_chunks"]
    uniform = sorted(((name, values[0]) for name, values in columns if len(set(values)) == 1),
                     key=lambda pair: order.index(pair[0]) if pair[0] in order else -1)

    header = ["Scale"] + [name for name, _ in varying]
    rows = [[str(s)] + [values[s] for _, values in varying] for s in range(len(pyramid))]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    table = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
             for row in (header, *rows)]

    # The table cannot be wrapped, so it alone may widen the banner; the prose wraps to fit.
    width = max(BANNER_WIDTH, max(len(row) for row in table) + 2)
    context = f"patch_size {patch_size}"
    if vae is not None:
        context += (f", {pyramid[0].shape[0]}ch VAE latents "
                    f"({vae.scale_factor}x smaller per side)")

    prose = [context, *lines]
    if uniform:
        prose.append("every scale: "
                     + "  ".join(f"{name}={value}" for name, value in uniform))
    wrapped = []
    for line in prose:
        # A continuation is indented under its own line, so it never reads as a new fact.
        indent = " " * (len("every scale: ") if line.startswith("every scale: ") else 2)
        wrapped += textwrap.wrap(line, width=width - 2, subsequent_indent=indent)
    report_setup(title, [*wrapped, *table])
