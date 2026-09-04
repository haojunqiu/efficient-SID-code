#!/usr/bin/env python
"""CLI entry point for retargeting: regenerate an image at a different resolution.

    python scripts/sample_retarget.py --config configs/retarget/pixel_exact_flash_attn.yaml \\
        image_path=path/to/image.png output_size="(186,372)" \\
        final_output_path=outputs/retarget.png

Key config overrides:
    seed          any integer; a different seed is a different sample
    output_size   required; the target (H, W) in pixels
    num_scales    "auto" (default), or an integer for the pyramid depth
"""
import time
_T0 = time.perf_counter()          # before `import torch`, so TOTAL matches the real wall clock

import torch

from efficient_sid import build
from efficient_sid.latent import maybe_load_vae, maybe_encode
from efficient_sid.timing import StageTimer
from efficient_sid.config import RetargetAppConfig, parse_args
from efficient_sid.utils import (
    seed_everything, torch_dtype, configure_matmul_precision,
    indexed_path, report_sample_progress,
    normalize_to_neg_one_to_one, load_image_as_tensor, torch_resize,
    save_image, save_preview,
)
from efficient_sid.applications.retarget import sample_retargeting


def main() -> None:
    config = parse_args(RetargetAppConfig)
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

    # --- VAE (optional) ---
    vae = maybe_load_vae(config.latent, device)

    # --- Load image ---
    image = load_image_as_tensor(config.image_path).to(dtype).to(device)
    if config.get("input_resize_to", None) is not None:
        image = torch_resize(image, build.parse_size(config.input_resize_to, "input_resize_to"))
    image = normalize_to_neg_one_to_one(image)
    image = maybe_encode(image, vae)

    assert config.output_size is not None, \
        "output_size is required for retargeting (e.g. output_size='(186,372)')"

    # --- Per-scale params + pyramid (from the original image — patches come from here) + one
    #     scheduler per scale (num_steps may differ per scale) ---
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

    # --- Build denoisers (from original image patches). The coarsest scale is taken straight from
    # the resized image and never denoised, so its denoiser is not built at all. ---
    denoisers = build.build_image_denoisers(
        config.image_denoiser,
        pyramid,
        skip_scales={num_scales - 1},
        dtype=dtype,
    )

    # --- Target resolution pyramid shapes ---
    sample_pyramid_shapes = pyramid_processor.gaussian_pyramid_shapes(
        (image.shape[0], *sample_size))

    build.report_scale_table(
        "Retarget", config, denoisers, pyramid, sample_pyramid_shapes, vae,
        lines=[f"scale {num_scales - 1} is fixed from the resized input, not denoised"],
    )

    # --- Resized image pyramid (for coarsest scale initialization) ---
    resized_img = torch_resize(image, sample_size)
    resized_pyramid = pyramid_processor.build_gaussian_pyramid(resized_img)
    resized_pyramid = [img.to(dtype) for img in resized_pyramid]

    # --- Run retargeting ---
    eta = config.scheduler.eta

    n = config.num_samples
    for i in range(n):
        seed = config.seed + i
        seed_everything(seed)
        result = sample_retargeting(
            denoisers=denoisers,
            schedulers=schedulers,
            pyramid_processor=pyramid_processor,
            sample_pyramid_shapes=sample_pyramid_shapes,
            resized_pyramid=resized_pyramid,
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

    timer.report_run("Retarget", denoisers=denoisers)


if __name__ == "__main__":
    main()
