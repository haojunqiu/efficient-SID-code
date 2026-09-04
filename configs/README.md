# Presets

One folder per application. Each holds `pixel_exact.yaml`, the reference: the exact backend,
which runs on any GPU, and the file whose header explains the knobs. The other presets in a
folder are that file with the denoiser backend switched, and each opens with one line saying so,
`Same as pixel_exact.yaml except image_denoiser.patch_denoiser.type: "..."`. The knob prose is
written once, in the base file.

| Preset | Backend | Notes |
|---|---|---|
| `pixel_exact` | `exact` | The reference. Materializes the `N×N` score matrix; on large inputs raise `query_chunks`, or use the next one. |
| `pixel_exact_flash_attn` | `exact_flash_attn` | Same math without the score matrix. Needs an Ampere or newer GPU. Faster when `d` is narrow, see below. |
| `pixel_ann` | `ann` | Approximate k-NN through FAISS. Requires faiss-gpu. Text style's Quickstart preset: about 3.5× faster than exact there. |
| `latent_exact` | `exact` in FLUX latent space | ~512 px and up. uncond, symmetric, tileable, structural analogy. |
| `latent_exact_flash_attn` | `exact_flash_attn` in latent space | Structural analogy only: its patch 3 gives `d = 144`, where the fused backend is 2 to 3× faster, see below. |
| `latent_ann` | `ann` in latent space | uncond only, for inputs so large that even the latent is big. |
| `uncond/gigapixel` | `auto` per scale | Multi-megapixel to gigapixel on a 96 GB GPU; see the README's Gigapixel section. |
| `uncond/paper_sinddm15` | `exact` | Frozen at the settings behind Table 1 of the paper. |

Retarget has no latent preset (untuned), and text style none at all: CLIP guidance would have to
backpropagate through the VAE.

## The fused-attention backend

`exact_flash_attn` runs the denoiser through PyTorch's fused attention, the memory-efficient
SDPA backend, via the homogeneous-coordinate trick of the paper's supplementary S3.2. The name
refers to the FlashAttention idea of never materializing the score matrix; it does not use the
flash-attn library. It needs an Ampere or newer GPU. `ann_flash_attn` applies the same kernel
to the `k` neighbours ANN retrieves.

**Memory.** `exact` materializes the `N×N` score matrix, so on a large image it needs
`query_chunks` raised until the slices fit; `exact_flash_attn` never materializes it and stays
within memory out of the box.

**Speed.** The gain depends on the patch vector `d = P²·C` (`C` = 3 in pixel space, 16 in the
FLUX latent) and on the GPU: the narrower `d`, the larger the share of the work it removes, and
how much that share is worth differs by GPU generation. The number of patches barely matters. Below `d ≈ 400` it is 2 to 3× faster or more. At the shipped pixel
patch 15, `d = 675`, we measured 1.8× on an RTX 6000 Ada, 1.1 to 1.3× on an RTX PRO 6000
Blackwell and about 1.0× on an RTX A6000: around this `d` the gain depends on the GPU generation.
Past `d ≈ 800`, the patch-7 latent presets at `d = 784` included, is where it starts to give no
gain or to run slower than `exact`. This is how the shipped presets chose: structural analogy's
patch-3 latent preset (`d = 144`) uses `exact_flash_attn`, the patch-7 latent presets use `exact`.

To see where your own GPU sits, run a `*_exact` preset as is and again with
`image_denoiser.patch_denoiser.type=exact_flash_attn`, and compare the `patch kernel` line of
the timing breakdown each run prints. Repeat at a few `patch_size` values to see how the gain
falls off with `d`: in pixel space patch 7, 11 and 15 give `d` = 147, 363 and 675; in latent
space patch 3, 5 and 7 give 144, 400 and 784. If your card still gains at `d = 784`, override
the latent presets with `image_denoiser.patch_denoiser.type=exact_flash_attn`.

## One-time costs

Two knobs pay a setup cost per input, listed separately in the timing breakdown: latent loads
the VAE and encodes the input, and ANN builds its index from the input's patches. Both are built
once per process, so `num_samples` amortizes them over many draws from one image. This is why
ANN can be slower than `exact` on a small input with one sample, and why it pays off for text
style, whose inversion makes hundreds of denoiser calls per image.

## Switching backends on the command line

Every preset accepts `image_denoiser.patch_denoiser.type=<backend>` as an override, with one
rule: start from a `*_exact` preset. The config class is chosen from the backend, and the exact
backends do not declare the five ANN keys (`k`, `index_type`, `nlist`, `nprobe`, `pq_m`), so an
ANN preset switched to an exact backend is rejected. An exact preset switched to `ann` loads,
with the ANN keys at their defaults.
