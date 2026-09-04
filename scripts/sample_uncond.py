#!/usr/bin/env python
"""CLI entry point for unconditional single-image generation.

    python scripts/sample_uncond.py --config configs/uncond/pixel_exact.yaml \\
        image_path=path/to/image.png final_output_path=outputs/sample.png

    # Large inputs (>= ~512px): sample in VAE latent space instead
    python scripts/sample_uncond.py --config configs/uncond/latent_exact.yaml \\
        image_path=path/to/image.png final_output_path=outputs/sample.png

The denoiser backend comes from the preset: pixel_exact.yaml, pixel_exact_flash_attn.yaml (same
maths, less memory), pixel_ann.yaml (approximate, needs faiss-gpu), latent_exact.yaml,
latent_ann.yaml; or override image_denoiser.patch_denoiser.type on the command line.

Key config overrides:
    seed                    any integer; a different seed is a different sample
    patch_size, num_scales  how large a patch of the input lands in the output
    num_steps, eta          sampling budget and stochasticity
"""
import time
_T0 = time.perf_counter()          # before `import torch`, so TOTAL matches the real wall clock

import torch

from efficient_sid import build
from efficient_sid.latent import maybe_load_vae, maybe_encode
from efficient_sid.timing import StageTimer
from efficient_sid.config import UncondAppConfig, parse_args
from efficient_sid.utils import (
    seed_everything, torch_dtype, configure_matmul_precision,
    indexed_path, report_sample_progress,
    normalize_to_neg_one_to_one, load_image_as_tensor, torch_resize,
    save_image, save_preview,
)
from efficient_sid.applications.uncond import sample_uncond


def main() -> None:
    config = parse_args(UncondAppConfig)
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

    # --- Load image (optionally VAE-encode to latent space) ---
    vae = maybe_load_vae(config.latent, device)
    image = load_image_as_tensor(config.image_path).to(dtype).to(device)
    if config.get("input_resize_to", None) is not None:
        image = torch_resize(image, build.parse_size(config.input_resize_to, "input_resize_to"))
    image = normalize_to_neg_one_to_one(image)
    image = maybe_encode(image, vae)

    # --- Pyramid + per-scale denoisers (one scheduler per scale; num_steps may differ per scale) ---
    sample_size = build.resolve_sample_size(config, image.shape[-2:], vae)
    config = build.resolve(
        config,
        image.shape[-2:],
        sample_size,
        auto_fields=build.AUTO_FIELDS,
    )
    schedulers = build.build_schedulers(config.scheduler, config.pyramid.num_scales, device)
    pyramid_processor, pyramid = build.build_pyramid(config.pyramid, image, dtype)
    denoisers = build.build_image_denoisers(config.image_denoiser, pyramid, dtype=dtype)

    # --- Output pyramid shapes (differ from the input's when output_size is set) ---
    sample_pyramid_shapes = pyramid_processor.gaussian_pyramid_shapes(
        (image.shape[0], *sample_size))

    build.report_scale_table(
        "Unconditional", config, denoisers, pyramid, sample_pyramid_shapes, vae)

    # --- Sample, then save ---
    n = config.num_samples
    for i in range(n):
        seed = config.seed + i
        seed_everything(seed)
        result = sample_uncond(
            denoisers=denoisers,
            schedulers=schedulers,
            pyramid_processor=pyramid_processor,
            sample_pyramid_shapes=sample_pyramid_shapes,
            eta=config.scheduler.eta,
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

    timer.report_run("Unconditional", denoisers=denoisers)


if __name__ == "__main__":
    main()
