#!/usr/bin/env python
"""CLI entry point for tileable (seamlessly tiling) texture generation.

    python scripts/sample_tileable.py --config configs/tileable/pixel_exact_flash_attn.yaml \\
        image_path=path/to/texture.png final_output_path=outputs/tile.png

    # Large inputs (>= ~512px): sample in VAE latent space instead
    python scripts/sample_tileable.py --config configs/tileable/latent_exact.yaml \\
        image_path=path/to/texture.png final_output_path=outputs/tile.png

Saves the tile plus 2x2 / 3x3 tiled grids next to it, and the same grids for the input tiled
naively, so the seams can be checked by eye.

Key config overrides:
    seed                any integer; a different seed is a different sample
    tiling.direction    "horizontal", "vertical", or "both"
    tiling.num_shifts   4 here, which adds the diagonal shift; 3 is the paper's setting
    num_steps, eta      sampling budget / stochasticity
"""
import time
_T0 = time.perf_counter()          # before `import torch`, so TOTAL matches the real wall clock

from pathlib import Path

import torch

from efficient_sid import build
from efficient_sid.latent import maybe_load_vae, maybe_encode
from efficient_sid.timing import StageTimer
from efficient_sid.config import TileableAppConfig, parse_args
from efficient_sid.utils import (
    seed_everything, torch_dtype, configure_matmul_precision,
    indexed_path, report_sample_progress,
    normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, load_image_as_tensor,
    torch_resize, save_image, save_preview, save_tiled_grids,
)
from efficient_sid.applications.tileable import sample_tileable, seam_error


def main() -> None:
    config = parse_args(TileableAppConfig)
    seed_everything(config.seed)
    configure_matmul_precision(
        config.precision.get("matmul"),
        config.precision.dtype,
        config.image_denoiser.patch_denoiser.bottleneck_dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(config.precision.dtype)

    diagnostics_config = config.get("diagnostics", {})
    timer = StageTimer(enabled=diagnostics_config.get("report_timing", True), t0=_T0)

    # --- VAE (optional latent-space acceleration for large images) ---
    vae = maybe_load_vae(config.latent, device)

    # --- Load image ---
    image = load_image_as_tensor(config.image_path).to(dtype).to(device)
    if config.get("input_resize_to", None) is not None:
        image = torch_resize(image, build.parse_size(config.input_resize_to, "input_resize_to"))
    image = normalize_to_neg_one_to_one(image)

    # Circular rolls work the same on a latent, so encode first and the whole pipeline runs in
    # latent space. pixel_image is the cropped input the latent was made from, which the seam-error
    # baseline and the saved input have to match.
    pixel_image = image if vae is None else vae.crop_to_multiple(image)
    image = maybe_encode(pixel_image, vae)

    # --- Per-scale params + pyramid + one scheduler per scale (num_steps may differ per scale) ---
    sample_size = build.resolve_sample_size(config, image.shape[-2:], vae)
    config = build.resolve(
        config,
        image.shape[-2:],
        sample_size,
        auto_fields=build.AUTO_FIELDS,
    )
    num_scales = config.pyramid.num_scales
    schedulers = build.build_schedulers(config.scheduler, num_scales, device)
    pyramid_processor, pyramid = build.build_pyramid(config.pyramid, image, dtype)
    sample_pyramid_shapes = pyramid_processor.gaussian_pyramid_shapes(
        (image.shape[0], *sample_size))

    # --- Tileable params ---
    tiling_direction = config.tiling.direction

    # --- Build denoisers (all scales — no skip for tileable) ---
    denoisers = build.build_image_denoisers(config.image_denoiser, pyramid, dtype=dtype)

    build.report_scale_table(
        "Tileable", config, denoisers, pyramid, sample_pyramid_shapes, vae,
        # Measured on the same [-1, 1] range as the result's seam error, so the two compare
        # directly.
        lines=[f"seam error of the input, tiled naively: "
               f"{seam_error(pixel_image, tiling_direction):.2f}/255"],
    )

    # --- Run tileable sampling ---
    eta = config.scheduler.eta

    n = config.num_samples
    for i in range(n):
        seed = config.seed + i
        seed_everything(seed)
        result = sample_tileable(
            denoisers=denoisers,
            schedulers=schedulers,
            pyramid_processor=pyramid_processor,
            sample_pyramid_shapes=sample_pyramid_shapes,
            tiling_config=config.tiling,
            eta=eta,
            vae=vae,
            device=device,
            dtype=dtype,
            intermediate_output_dir=indexed_path(
                diagnostics_config.get("intermediate_output_dir"), i, n),
        )
        out_path = indexed_path(config.final_output_path, i, n)
        if out_path is not None:
            save_image(result, out_path)
            save_preview(result, out_path, config.get("preview_long_side"))
            save_tiled_grids(result, out_path, tiling_direction, "sample")
        report_sample_progress(i, n, seed, out_path)

    # The same grids for the input, tiled naively. Written once: the input does not change.
    if config.final_output_path is not None:
        out = Path(config.final_output_path)
        input_image = unnormalize_to_zero_to_one(pixel_image)
        save_image(input_image, out.with_name(f"{out.stem}_input{out.suffix}"))
        save_tiled_grids(input_image, config.final_output_path, tiling_direction, "input")

    timer.report_run("Tileable", denoisers=denoisers)


if __name__ == "__main__":
    main()
