#!/usr/bin/env python
"""CLI entry point for symmetric image generation.

    python scripts/sample_symmetric.py --config configs/symmetric/pixel_exact_flash_attn.yaml \\
        image_path=path/to/image.png symmetry.axis=horizontal final_output_path=outputs/sym.png

    # Large inputs (>= ~512px): sample in VAE latent space instead
    python scripts/sample_symmetric.py --config configs/symmetric/latent_exact.yaml \\
        image_path=path/to/image.png symmetry.axis=horizontal final_output_path=outputs/sym.png

Key config overrides:
    seed                     any integer; a different seed is a different sample
    symmetry.axis            "horizontal" or "vertical"
    symmetry.mirror_source   "left"/"right" or "top"/"bottom" -- which half to keep
"""
import time
_T0 = time.perf_counter()          # before `import torch`, so TOTAL matches the real wall clock

import warnings

import torch

from efficient_sid import build
from efficient_sid.latent import maybe_load_vae, maybe_encode
from efficient_sid.timing import StageTimer
from efficient_sid.config import SymmetricAppConfig, parse_args
from efficient_sid.utils import (
    seed_everything, torch_dtype, configure_matmul_precision,
    indexed_path, report_sample_progress,
    normalize_to_neg_one_to_one, load_image_as_tensor, torch_resize,
    save_image, save_preview,
)
from efficient_sid.applications.symmetric import SymmetryConfig, sample_symmetric


def main() -> None:
    config = parse_args(SymmetricAppConfig)
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

    # Mirroring is a plain spatial flip, so it works the same on a latent: encode first and the
    # whole pipeline runs in latent space.
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

    # --- Symmetric params ---
    symmetry = config.symmetry.axis
    mirror_source = config.symmetry.mirror_source
    # Which halves are legal depends on the axis, which no field type can express. An axis that is
    # neither falls through to the check in applications.symmetric._get_sym_fn.
    legal_sources = {"horizontal": ("left", "right"), "vertical": ("top", "bottom")}.get(symmetry)
    if legal_sources is not None and mirror_source not in legal_sources:
        warnings.warn(
            f"symmetry.mirror_source={mirror_source!r} is not one of "
            f"{', '.join(legal_sources)} for symmetry.axis={symmetry!r}; "
            f"using {legal_sources[0]!r}.")
        mirror_source = legal_sources[0]
    symmetry_config = SymmetryConfig(axis=symmetry, mirror_source=mirror_source)

    # --- Build denoisers (every scale is denoised) ---
    denoisers = build.build_image_denoisers(
        config.image_denoiser,
        pyramid,
        dtype=dtype,
    )

    build.report_scale_table(
        "Symmetric", config, denoisers, pyramid, sample_pyramid_shapes, vae)

    # --- Run symmetric sampling ---
    eta = config.scheduler.eta

    n = config.num_samples
    for i in range(n):
        seed = config.seed + i
        seed_everything(seed)
        result = sample_symmetric(
            denoisers=denoisers,
            schedulers=schedulers,
            pyramid_processor=pyramid_processor,
            sample_pyramid_shapes=sample_pyramid_shapes,
            symmetry_config=symmetry_config,
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
        report_sample_progress(i, n, seed, out_path)

    timer.report_run("Symmetric", denoisers=denoisers)


if __name__ == "__main__":
    main()
