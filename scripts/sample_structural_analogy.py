#!/usr/bin/env python
"""CLI entry point for structural analogy: put the style image's texture on the structure
image's layout.

    python scripts/sample_structural_analogy.py --config configs/structural_analogy/pixel_exact.yaml \\
        structure_image_path=examples/structural_analogy/content/s_char.jpg \\
        style_image_path=examples/structural_analogy/style/duck_mosaic.jpg \\
        final_output_path=outputs/analogy.png

    # Large inputs (>= ~512px): sample in VAE latent space instead
    python scripts/sample_structural_analogy.py --config configs/structural_analogy/latent_exact.yaml \\
        structure_image_path=... style_image_path=... final_output_path=outputs/analogy.png

Key config overrides:
    seed                    any integer; a different seed is a different sample
    noise_level             how far to invert the structure image; null = full inversion
    num_steps               a scalar, or a per-scale list finest -> coarsest, e.g. "[10,10,50]"
    num_scales, patch_size  together they set how large a style patch lands in the output --
                            the two the result is most sensitive to
"""
import time
_T0 = time.perf_counter()          # before `import torch`, so TOTAL matches the real wall clock

import torch

from efficient_sid import build
from efficient_sid.latent import maybe_load_vae, maybe_encode
from efficient_sid.timing import StageTimer
from efficient_sid.config import StructuralAnalogyAppConfig, parse_args
from efficient_sid.utils import (
    seed_everything, torch_dtype, configure_matmul_precision,
    indexed_path, report_sample_progress,
    normalize_to_neg_one_to_one, load_image_as_tensor, resize_to_short_side,
    save_image, save_preview,
)
from efficient_sid.applications.structural_analogy import sample_structural_analogy


def main() -> None:
    config = parse_args(StructuralAnalogyAppConfig)
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

    # --- Load style image (supplies texture/prior at every scale) ---
    style_image = load_image_as_tensor(config.style_image_path).to(dtype).to(device)

    # --- Load structure image (supplies coarse layout, inverted at coarsest scale only) ---
    structure_image = load_image_as_tensor(config.structure_image_path).to(dtype).to(device)

    # --- Resolution alignment (pixel space, before any VAE encode) ---
    # Both images go to the same short side, so a patch covers the same fraction of each.
    # ``output_short_side`` sets it; null takes the style image's short side.
    output_short_side = config.get("output_short_side") or min(style_image.shape[-2:])
    style_image = resize_to_short_side(style_image, output_short_side)
    structure_image = resize_to_short_side(structure_image, output_short_side)

    style_image = normalize_to_neg_one_to_one(style_image)
    structure_image = normalize_to_neg_one_to_one(structure_image)

    # Encode each to the shared VAE latent space (independently -- they need not share shape).
    # From here on style_image/structure_image are latents; all pyramids/denoisers/sampling
    # operate in latent space, and the final result is decoded back to pixels on save.
    style_image = maybe_encode(style_image, vae)
    structure_image = maybe_encode(structure_image, vae)

    # --- Per-scale params + one scheduler per scale (num_steps can differ per scale, e.g. fewer
    #     steps on the finer scales that mostly refine/blend rather than drive the style transfer) ---
    # Patches come from the style image; the output is sampled on the structure image's grid, which
    # output_short_side already fixed above. num_scales is left out of auto_fields: it sets how large
    # a style patch lands in the output, so it is a knob to sweep by eye rather than one to derive.
    config = build.resolve(
        config,
        style_image.shape[-2:],
        structure_image.shape[-2:],
        auto_fields=(
            "image_denoiser.patch_denoiser.type",
            "image_denoiser.query_stride",
            "image_denoiser.patch_denoiser.query_chunks",
        ),
    )
    num_scales = config.pyramid.num_scales
    schedulers = build.build_schedulers(config.scheduler, num_scales, device)

    s_coarsest = num_scales - 1

    # Inversion happens only at the coarsest scale, and uses that scale's scheduler.

    # --- Build style pyramid + denoisers at every scale ---
    pyramid_processor, style_pyramid = build.build_pyramid(config.pyramid, style_image, dtype)
    style_denoisers = build.build_image_denoisers(config.image_denoiser, style_pyramid, dtype=dtype)

    # --- Build structure pyramid (reuse the style processor — same config); only the coarsest-scale
    #     denoiser is needed, and it is built with the inversion scheduler ---
    structure_pyramid = [img.to(dtype)
                         for img in pyramid_processor.build_gaussian_pyramid(structure_image)]
    structure_coarsest_denoiser = build.build_image_denoiser(
        build.image_denoiser_config_at_scale(config.image_denoiser, s_coarsest),
        structure_pyramid[s_coarsest],
        dtype,
    )
    structure_coarsest_image = structure_pyramid[s_coarsest].unsqueeze(0).to(dtype).to(device)

    # Output is sampled on the *structure*'s pyramid grid (content geometry); style denoisers
    # supply texture at each scale regardless of their own (style) resolution.
    sample_pyramid_shapes = pyramid_processor.gaussian_pyramid_shapes(structure_image.shape)

    build.report_scale_table(
        "Structural analogy", config, style_denoisers, style_pyramid, sample_pyramid_shapes, vae,
        lines=[f"scale {s_coarsest} inverted from the structure image at "
               f"noise_level {config.get('noise_level')}"],
    )

    # --- Run structural analogy ---
    eta = config.scheduler.eta
    noise_level = config.get("noise_level", None)

    # Opt-in: set intermediate_output_dir to dump the per-scale results and the inverted noise.
    intermediate_output_dir = diagnostics_config.get("intermediate_output_dir")

    n = config.num_samples
    for i in range(n):
        seed = config.seed + i
        seed_everything(seed)
        result = sample_structural_analogy(
            style_denoisers=style_denoisers,
            schedulers=schedulers,
            pyramid_processor=pyramid_processor,
            sample_pyramid_shapes=sample_pyramid_shapes,
            structure_coarsest_denoiser=structure_coarsest_denoiser,
            structure_coarsest_image=structure_coarsest_image,
            noise_level=noise_level,
            eta=eta,
            vae=vae,
            device=device,
            dtype=dtype,
            intermediate_output_dir=indexed_path(intermediate_output_dir, i, n),
        )
        out_path = indexed_path(config.final_output_path, i, n)
        if out_path is not None:
            save_image(result, out_path)
            save_preview(result, out_path, config.get("preview_long_side"))
        report_sample_progress(i, n, seed, out_path)

    timer.report_run("Structural analogy", denoisers=style_denoisers)


if __name__ == "__main__":
    main()
