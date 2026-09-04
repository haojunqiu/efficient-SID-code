"""Shared utilities: image tensor ops, image I/O, seeding, and numeric precision."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Lift PIL's ~179 MP decompression-bomb guard: gigapixel inputs (up to ~1.2 GP, e.g. the tokyo/moon
# presets) are the point here, and are loaded from local files the user chose -- not untrusted
# network data the guard is meant to protect against.
Image.MAX_IMAGE_PIXELS = None

from efficient_sid.latent import FluxVAE, maybe_decode
from efficient_sid.timing import record_io


# ---------------------------------------------------------------------------
# Image tensor ops -- value range and geometry
# ---------------------------------------------------------------------------

def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) * 0.5


def torch_resize(image: torch.Tensor, size: Sequence[int], mode: str = 'bicubic') -> torch.Tensor:
    """Resize a ``(C, H, W)`` image to ``size``, an ``(H, W)`` pair.

    For resizes outside the algorithm; pyramid building and Laplacian blending use
    ``resize_right.resize``. Any value range.
    """
    return F.interpolate(
        image.unsqueeze(0),
        size=tuple(size),
        mode=mode,
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def resize_to_short_side(image: torch.Tensor, target_short_side: int) -> torch.Tensor:
    """Resize a ``(C, H, W)`` image so its shorter side equals ``target_short_side``, preserving
    aspect ratio. Returns the input unchanged when the short side already matches."""
    H, W = image.shape[-2:]
    short = min(H, W)
    if short == target_short_side:
        return image
    scale = target_short_side / short
    return torch_resize(image, (round(H * scale), round(W * scale)))


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image_as_tensor(image_path: Union[str, Path]) -> torch.Tensor:
    """Load an image file as a ``(C, H, W)`` float tensor in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().float().div(255)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``(C, H, W)`` float tensor in [0, 1] to a PIL image."""
    return Image.fromarray(
        tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).detach().cpu().numpy()
    )


def save_image(image: torch.Tensor, filename: Union[str, Path], quality: int = 90) -> None:
    """Save a (C, H, W) float tensor in [0, 1] as an image file.

    The format follows ``filename``'s extension. For JPEG (``.jpg`` / ``.jpeg``), ``quality``
    (default 90) is applied with ``optimize``/``progressive``; it is ignored for PNG, which is
    lossless. Prefer JPEG for very large outputs -- far smaller on disk and much faster to encode.

    JPEG cannot encode a side longer than 65500 px (a libjpeg limit); a result that big -- a very
    wide retarget, say -- falls back to PNG.

    Wrapped in ``timing.record_io`` (GPU->CPU copy + encode + disk write) so this time is reported
    on its own and kept out of the sampling stage -- see ``timing.timed_compute``.
    """
    with record_io():
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        pil_image = tensor_to_pil(image)
        is_jpeg = filename.suffix.lower() in (".jpg", ".jpeg")
        if is_jpeg and max(pil_image.size) > 65500:
            png = filename.with_suffix(".png")
            print(f"  {filename.name} is {pil_image.size[0]}x{pil_image.size[1]}, past JPEG's "
                  f"65500 px limit; writing {png.name} instead.")
            filename, is_jpeg = png, False
        save_kwargs = dict(quality=quality, optimize=True, progressive=True) if is_jpeg else {}
        pil_image.save(filename, **save_kwargs)


def save_preview(
    result: torch.Tensor,
    final_output_path: Union[str, Path],
    long_side: Optional[int],
) -> None:
    """Save a downsampled JPEG preview of ``result`` alongside ``final_output_path``.

    A no-op when ``long_side`` is None, so callers can invoke it unconditionally. ``result`` is the
    same (C, H, W) tensor in [0, 1] the full-size save gets, so the preview costs one in-memory
    resize.

    Written as ``<name>_preview.jpg``: a preview wants to open instantly, so it is always JPEG.
    """
    if long_side is None:
        return
    _, H, W = result.shape
    if max(H, W) > long_side:
        scale = long_side / max(H, W)
        result = torch_resize(result.float(), (round(H * scale), round(W * scale)))
    out = Path(final_output_path)
    save_image(result, out.with_name(f"{out.stem}_preview.jpg"))


def _tile_image_grid(img: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    """Repeat a ``[C, H, W]`` image into a ``(rows, cols)`` grid, giving ``[C, H*rows, W*cols]``."""
    rows, cols = grid_size
    return img.repeat(1, rows, cols)


def save_tiled_grids(
    img: torch.Tensor,
    final_output_path: Union[str, Path],
    tiling_direction: str,
    label: str,
) -> None:
    """Write ``img`` repeated into tiled grids alongside ``final_output_path``.

    ``tiling_direction`` picks the grids: 1x2 and 1x3 across for 'horizontal', 2x1 and 3x1 down
    for 'vertical', 2x2 and 3x3 for 'both'.

    Written as ``<stem>_<label>_tiled_<rows>x<cols>``, so ``out/tile.png`` with label ``sample``
    gives ``out/tile_sample_tiled_2x2.png``. ``img`` is any [C, H, W] image in [0, 1] -- the
    sample or the input it was drawn from -- and ``label`` keeps the two sets apart.
    """
    out = Path(final_output_path)
    grids = {
        "horizontal": [(1, 2), (1, 3)],
        "vertical": [(2, 1), (3, 1)],
        "both": [(2, 2), (3, 3)],
    }[tiling_direction]
    for rows, cols in grids:
        save_image(
            _tile_image_grid(img, (rows, cols)),
            out.with_name(f"{out.stem}_{label}_tiled_{rows}x{cols}{out.suffix}"),
        )


def save_scale_result(
    x: torch.Tensor,
    scale: int,
    output_dir: Optional[Union[str, Path]],
    vae: Optional[FluxVAE] = None,
) -> None:
    """Save the finished result of one pyramid scale: the coarse-to-fine trace.

    A no-op when ``output_dir`` is None (the default), so every sampler can call this
    unconditionally at the end of a scale.

    Intermediates are per-scale, not per-timestep -- a per-step dump would be hundreds of
    near-noise images. When a ``vae`` is given the latent is decoded first; the decoder is fully
    convolutional, so coarser scales decode to smaller images.
    """
    if output_dir is None:
        return
    save_image(
        unnormalize_to_zero_to_one(maybe_decode(x, vae)).squeeze(0),
        Path(output_dir) / f"scale{scale}_result.png",
    )


# ---------------------------------------------------------------------------
# Drawing several samples from one image
# ---------------------------------------------------------------------------

def indexed_path(
    path: Optional[Union[str, Path]],
    i: int,
    n: int,
) -> Optional[Union[str, Path]]:
    """Return the path for sample ``i`` of ``n``, unchanged when ``n`` is 1::

        out/sample.png  ->  out/sample_000.png, out/sample_001.png, ...

    A directory has no suffix, so this serves ``intermediate_output_dir`` too.
    """
    if path is None or n == 1:
        return path
    p = Path(path)
    return p.with_name(f"{p.stem}_{i:03d}{p.suffix}")


def report_sample_progress(i: int, n: int, seed: int, path: Optional[Union[str, Path]]) -> None:
    """Print one line when sample ``i`` finishes; silent for a single sample.

    The counter starts at 1 while the filename starts at 0: sample ``i`` uses ``seed + i``, so
    sample 0 has to use the configured seed.
    """
    if n == 1:
        return
    where = f"  ->  {path}" if path is not None else ""
    print(f"  [{i + 1}/{n}]  seed {seed}{where}")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Numeric precision
# ---------------------------------------------------------------------------

@dataclass
class PrecisionConfig:
    """What the run's tensors are. The patch denoiser's own compute dtype is not here -- it is
    ``image_denoiser.patch_denoiser.bottleneck_dtype``, read by that one class."""
    #: "float32" | "float16" | "bfloat16" -- the tensors the run works in.
    dtype: str = "bfloat16"
    #: "highest" (float32) | "high" (TF32) | "medium" (bf16). Affects float32 only.
    matmul: str = "highest"


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def torch_dtype(name: Optional[str]) -> torch.dtype:
    """Return the torch dtype a config names.

    Matmul precision (TF32 etc.) is a separate axis, set by ``configure_matmul_precision``.
    """
    s = (name or "float32").lower()
    if s not in _DTYPES:
        raise ValueError(f"Unknown dtype {s!r}; expected one of {', '.join(_DTYPES)}")
    return _DTYPES[s]


_MATMUL_PRECISIONS = ("highest", "high", "medium")


def configure_matmul_precision(matmul: Optional[str], *dtypes: str) -> None:
    """Set torch's process-wide float32 matmul precision.

    ``highest`` (the default) keeps true float32; ``high`` runs float32 matmuls on TF32 tensor
    cores; ``medium`` accumulates in bfloat16. This affects *only* float32 matmuls, so it is inert
    when ``dtypes`` names no float32 (e.g. ``dtype: bfloat16``, the shipped default), and says so.
    """
    matmul = (matmul or "highest").lower()
    if matmul not in _MATMUL_PRECISIONS:
        raise ValueError(
            f"Unknown matmul precision {matmul!r}; expected one of {', '.join(_MATMUL_PRECISIONS)}")
    torch.set_float32_matmul_precision(matmul)
    if matmul != "highest":
        has_fp32 = any((d or "float32").lower() == "float32" for d in dtypes)
        label = {"high": "high (TF32 tensor cores)",
                 "medium": "medium (bfloat16 accumulate)"}[matmul]
        note = "" if has_fp32 else " -- no float32 tensors this run, so no effect"
        print(f"  matmul precision: {label}{note}")
