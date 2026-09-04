# efficient-sid

**Efficient and Training-Free Single-Image Diffusion Models** (CVPR 2026)
Haojun Qiu, Kiriakos N. Kutulakos, David B. Lindell · [Project page](https://haojunqiu.github.io/efficient-SID/)

**A diffusion prior from a single image: no training, no neural network.**

The optimal denoiser for a dataset of image patches has a closed form: a weighted average of those
patches. We use the patches of a single image as that dataset, so the denoiser is computed directly
from them, with no training and no network: **the image is the model.**

Unconditional generation samples from that prior. The other five applications are variants of the
same sampler with an extra constraint, guidance term or initialization:

| Application | What it does |
|---|---|
| **Unconditional** | Generate new images with the input's patch statistics |
| **Retargeting** | Rebuild the image at a new resolution or aspect ratio from its own patches |
| **Symmetric** | Generate horizontally or vertically symmetric variants |
| **Tileable** | Generate seamlessly tiling textures |
| **Structural analogy** | Transfer one image's texture onto another's layout |
| **Text-driven style** | Restyle an image toward a text prompt via CLIP guidance |

Computed exactly, the denoiser weights every clean patch of the image by its distance to the noisy
patch being denoised: an `N×N` distance matrix per step. Three accelerations cut that down: a fused
kernel borrowed from attention, which has the same computation structure and never materializes the
matrix; approximate nearest neighbours, which keep only the `k` largest weights; and sampling in a
VAE latent space, where `N` is 64× smaller. See [Making it fast](#making-it-fast).

## Install

Requires a CUDA GPU with an NVIDIA driver that supports CUDA 12 (driver 525 or newer).

```bash
conda env create -f environment.yml    # Python 3.10, PyTorch 2.8, faiss-gpu, diffusers, CLIP
conda activate efficient-sid
pip install -e .
```

This installs everything the commands below need, at the exact versions the code is tested
with. Use a recent conda or [Miniforge](https://github.com/conda-forge/miniforge); the classic
solver of conda 4.x takes very long on this file.

Two paths download weights on first use. Text-driven style fetches CLIP, with no account needed.
The latent presets fetch the autoencoder of
[`FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (Apache-2.0), which
Hugging Face gates behind a one-time acceptance of the model's terms:

1. Create a free [Hugging Face account](https://huggingface.co/join) if you have none.
2. Open the [model page](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and click
   **Agree and access repository**. Access is granted immediately.
3. Create a read token at [Settings → Access Tokens](https://huggingface.co/settings/tokens),
   then run `hf auth login` once and paste it (or set `HF_TOKEN=<token>` in your shell).

Without this, every `latent_*` preset fails with a 401 on its first run; the pixel presets are
unaffected.

## Quickstart

Every application is one script, `scripts/sample_<app>.py`, run with one preset,
`--config configs/<app>/<preset>.yaml`, and any config key overridable as `key=value` on the
command line. All commands below run as-is on the images in [`examples/`](examples/). `final_output_path` is the
file the result is written to; its directory is created for you.

```bash
# Unconditional: a new image with the input's patch statistics
python scripts/sample_uncond.py --config configs/uncond/pixel_exact.yaml \
    image_path=examples/uncond/balloons.png final_output_path=outputs/uncond/balloons.png

# Retarget to a wider aspect ratio
python scripts/sample_retarget.py --config configs/retarget/pixel_exact.yaml \
    image_path=examples/retarget/fruit.png output_size="(224,448)" \
    final_output_path=outputs/retarget/fruit_wide.png

# Symmetric
python scripts/sample_symmetric.py --config configs/symmetric/pixel_exact.yaml \
    image_path=examples/symmetric/tapestry.png symmetry.axis=horizontal seed=11 \
    final_output_path=outputs/symmetric/tapestry.png

# Seamlessly tileable texture. Also writes 2x2 / 3x3 grids of the sample and of the naively tiled input.
python scripts/sample_tileable.py --config configs/tileable/pixel_exact.yaml \
    image_path=examples/tileable/bricks.png final_output_path=outputs/tileable/bricks.png

# Structural analogy: the style image's texture on the structure image's layout
python scripts/sample_structural_analogy.py --config configs/structural_analogy/pixel_exact.yaml \
    structure_image_path=examples/structural_analogy/content/s_char.jpg \
    style_image_path=examples/structural_analogy/style/duck_mosaic.jpg seed=11 \
    final_output_path=outputs/structural_analogy/s_char_duck.png

# Text-driven style (CLIP-guided; no style image)
python scripts/sample_text_style.py --config configs/text_style/pixel_ann.yaml \
    image_path=examples/text_style/bagan.png text="Van Gogh style" \
    final_output_path=outputs/text_style/bagan_vangogh.png
```

Megapixel inputs sample in the FLUX VAE latent space instead (see [Making it fast](#making-it-fast)).
The first latent run downloads the autoencoder; see [Install](#install) for the Hugging Face login it needs.

```bash
python scripts/sample_uncond.py --config configs/uncond/latent_exact.yaml \
    image_path=examples/uncond/succulent.png final_output_path=outputs/uncond/succulent.png

python scripts/sample_tileable.py --config configs/tileable/latent_exact.yaml \
    image_path=examples/tileable/crinkled_paper.png final_output_path=outputs/tileable/crinkled_paper.png

python scripts/sample_structural_analogy.py --config configs/structural_analogy/latent_exact.yaml \
    structure_image_path=examples/structural_analogy/content/mimi.jpg \
    style_image_path=examples/structural_analogy/style/fur.jpg \
    final_output_path=outputs/structural_analogy/mimi_fur.png
```

- **Presets.** Each `configs/<app>/` folder holds a few, named by the space and the denoiser
  backend; [configs/README.md](configs/README.md) lists them and says when to use which.
- **Several samples.** `num_samples=20` writes `balloons_000.png` … `balloons_019.png` from one
  process, reusing the pyramid and denoisers; sample *i* uses `seed + i`.
- **Intermediate scales.** `diagnostics.intermediate_output_dir=<dir>` saves each pyramid
  scale's result.
- **Tileable shifts: differs from the paper.** The presets use `tiling.num_shifts: 4`; **the
  paper used 3.** The fourth, diagonal shift also covers the tile's corners, closing a seam that
  can otherwise appear there. `tiling.num_shifts=3` gives the variant presented in the paper.

The scripts are thin wrappers over the `efficient_sid` package; `scripts/sample_uncond.py` is the
shortest end-to-end example of using it as a library.

## Making it fast

Computed exactly, one denoising step needs the distance between every noisy patch and every clean
patch of the input, an `N×N` matrix: quadratic in the number of patches, and the reason the paper
has three accelerations. They are independent knobs and combine freely.
The backend is `image_denoiser.patch_denoiser.type`: `exact` (the reference), `exact_flash_attn`,
`ann`, `ann_flash_attn`.

| Acceleration | Turn on with | What it does | Use when |
|---|---|---|---|
| **Fused attention** | `type: exact_flash_attn` (or `ann_flash_attn`) | The same denoiser without materializing the `N×N` score matrix: memory-lean out of the box, and faster when the patch vector is narrow. | Depends on `d = P²·C` and your GPU; see [configs/README.md](configs/README.md). |
| **Approximate k-NN** | `type: ann` (or `ann_flash_attn`) | Weights the `k` nearest patches (FAISS) instead of all `N`: `O(N^1.5)` at the shipped `nlist = sqrt(N)`. | Large pixel-space inputs, or many samples from one input (`num_samples`), where its one-time index build is amortized; on a small image with a single sample the build may not be worth it. |
| **Latent space** | `latent.enabled: true` | Samples in the FLUX VAE latent, 8× smaller per side, so ~64× fewer patches. | ~512 px and up; what makes megapixel and gigapixel inputs feasible. |

Every app ships `pixel_exact` (the reference, runs on any GPU), `pixel_exact_flash_attn`,
`pixel_ann` and, where it pays off, `latent_exact`; each preset file says how it differs from
`pixel_exact`. Which backend is fastest depends on the patch vector `d` and on your GPU:
[configs/README.md](configs/README.md) has the measured rule and how to check it on your own card.

`exact` and `exact_flash_attn` are the same math: in float32 their outputs agree to within a grey
level. In the shipped `bfloat16` their rounding differs enough that the same seed may lead to a
visibly different image.

Every run prints a wall-time breakdown (VAE, per-scale denoising, decode, image writing);
`diagnostics.report_timing=false` silences it.

## Datasets

`examples/` is committed, so Quickstart runs straight after a clone. The larger sets behind the
paper's figures and tables, and more images to try beyond `examples/`, are distributed as a release
and land in `datasets/`, which is gitignored:

```bash
python datasets/download_data.py --status            # list the sets and which are already downloaded
python datasets/download_data.py uncond/sinddm15     # one set (the Table 1 benchmark, 1.5 MB)
python datasets/download_data.py --all               # all eleven sets, about 450 MB
```

Files already in place are skipped. The Table 1 metrics were produced with
[`configs/uncond/paper_sinddm15.yaml`](configs/uncond/paper_sinddm15.yaml), frozen at that run's
settings, on the `uncond/sinddm15` set. The images are third-party works under their own
non-commercial licences; [datasets/README.md](datasets/README.md) carries the credits those terms
require.

## Gigapixel

Unconditional generation on multi-megapixel to gigapixel inputs, in latent space.
[`configs/uncond/gigapixel.yaml`](configs/uncond/gigapixel.yaml) fits a 96 GB GPU and derives its
per-scale parameters from the input's size, so a new image needs only an `image_path`. `eta` and
`fold_rho` are the quality knobs (see the config header). The three paper inputs:

```bash
python datasets/download_data.py uncond/gigapixel     # 394 MB; credits in datasets/README.md

# Moon, widened 2x; eta 0.0 for fine local detail
python scripts/sample_uncond.py --config configs/uncond/gigapixel.yaml \
    image_path=datasets/uncond/gigapixel/moon.jpg scheduler.eta=0.0 \
    output_size="(18192, 54576)" image_denoiser.patch_denoiser.nprobe=40 final_output_path=outputs/gigapixel/moon.jpg

# Tokyo, native size; eta 1.0 for global coherence
python scripts/sample_uncond.py --config configs/uncond/gigapixel.yaml \
    image_path=datasets/uncond/gigapixel/tokyo.jpg scheduler.eta=1.0 \
    output_size=null image_denoiser.patch_denoiser.nprobe=40 final_output_path=outputs/gigapixel/tokyo.jpg

# Painting, widened ~3.3x
python scripts/sample_uncond.py --config configs/uncond/gigapixel.yaml \
    image_path=datasets/uncond/gigapixel/painting.jpg scheduler.eta=1.0 \
    output_size="(14336, 70080)" image_denoiser.patch_denoiser.nprobe=20 final_output_path=outputs/gigapixel/painting.jpg
```

## Citation

```bibtex
@InProceedings{Qiu_2026_CVPR,
    author    = {Qiu, Haojun and Kutulakos, Kiriakos N. and Lindell, David B.},
    title     = {Efficient and Training-Free Single-Image Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {36157-36167}
}
```

## License

MIT, see [LICENSE](LICENSE). Vendors MIT-licensed code from
[ResizeRight](https://github.com/assafshocher/ResizeRight) (image pyramids) and
[Text2LIVE](https://github.com/omerbt/Text2LIVE) (CLIP guidance).
