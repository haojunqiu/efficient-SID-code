#!/usr/bin/env python
"""CLI entry point for text-driven style transfer: restyle an image toward a text prompt.

There is no style image -- the prompt supplies the style, via CLIP guidance.

    python scripts/sample_text_style.py --config configs/text_style/pixel_ann.yaml \\
        image_path=examples/text_style/bagan.png text="Van Gogh style" \\
        final_output_path=outputs/vangogh.png

Key config overrides:
    seed                       any integer; a different seed is a different sample
    text                       the guiding text prompt (e.g. "Van Gogh style")
    noise_level                how far to invert the content image first, 0 to 1; 0.3
    clip_guidance.strength     guidance step size gamma; 0.1 to 1.0 all stable, no hard cap; 1.0
    clip_guidance.fill_factor  fraction of pixels guided each step (0.1-1.0); 0.5
    clip_guidance.llambda      weight on the new estimate, 0 to 1; lower keeps more momentum; 0.1
    patch_size, num_steps      7 and 200 here
"""
import time
_T0 = time.perf_counter()          # before `import torch`, so TOTAL matches the real wall clock

import torch
from omegaconf import OmegaConf

from efficient_sid import build
from efficient_sid.scheduler import make_scheduler
from efficient_sid.timing import StageTimer
from efficient_sid.config import TextStyleAppConfig, parse_args
from efficient_sid.utils import (
    seed_everything, torch_dtype, configure_matmul_precision,
    indexed_path, report_sample_progress,
    normalize_to_neg_one_to_one, load_image_as_tensor, torch_resize,
    save_image, save_preview,
)
from efficient_sid.clip_extractor import get_clip_extractor
from efficient_sid.applications.text_style import sample_text_style


def main() -> None:
    config = parse_args(TextStyleAppConfig)
    # There is no pyramid here, so a num_scales override cannot be honoured. Reject it instead of
    # ignoring it -- a stale config would otherwise silently sample at one scale. A leftover
    # num_scales: 1 is accepted, since that is what this app does anyway.
    num_scales = OmegaConf.select(config, "pyramid.num_scales", default=1)
    if num_scales != 1:
        raise ValueError(
            f"pyramid.num_scales={num_scales}: text_style is single-scale by design and has no "
            "pyramid (see efficient_sid/applications/text_style.py). Drop the key to proceed.")
    # The remaining size-driven fields are read straight off the config below, so an "auto" left in
    # one would reach the denoiser unresolved.
    build.reject_auto(config)
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

    # --- Load content image (the single image being restyled; supplies both the patch prior
    #     and the layout that inversion preserves) ---
    content_image = load_image_as_tensor(config.image_path).to(dtype).to(device)
    if config.get("input_resize_to", None) is not None:
        content_image = torch_resize(content_image, build.parse_size(config.input_resize_to, "input_resize_to"))
    content_image = normalize_to_neg_one_to_one(content_image)

    # --- One scheduler + one denoiser, both at the content image's own resolution ---
    scheduler = make_scheduler(config.scheduler, device)
    content_denoiser = build.build_image_denoiser(
        config.image_denoiser,
        content_image,
        dtype,
    )

    # --- CLIP extractor (augmented CLIP image/text embedder) ---
    clip_extractor = get_clip_extractor(
        config.clip_extractor,
        device=str(device),
    )

    build.report_setup("Text style", [
        f"content image {content_image.shape[-2]}x{content_image.shape[-1]} (HxW), single-scale, "
        f"patch_size {config.image_denoiser.patch_size}",
        f"patch dataset: {content_denoiser.patch_denoiser.N} patches",
    ])

    # --- Run text style transfer ---
    # Opt-in: set intermediate_output_dir to dump the per-step guided results.
    n = config.num_samples
    for i in range(n):
        seed = config.seed + i
        seed_everything(seed)
        result = sample_text_style(
            content_denoiser=content_denoiser,
            scheduler=scheduler,
            clip_extractor=clip_extractor,
            content_image=content_image.unsqueeze(0),
            text=config.text,
            clip_config=config.clip_guidance,
            noise_level=config.get("noise_level", 0.3),
            eta=config.scheduler.eta,
            dtype=dtype,
            intermediate_output_dir=indexed_path(
                diagnostics_config.get("intermediate_output_dir"), i, n),
        )
        out_path = indexed_path(config.final_output_path, i, n)
        if out_path is not None:
            save_image(result, out_path)
            save_preview(result, out_path, config.get("preview_long_side"))
        report_sample_progress(i, n, seed, out_path)

    timer.report_run("Text style", denoisers=[content_denoiser])


if __name__ == "__main__":
    main()
