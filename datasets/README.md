# Datasets

The repository commits only `examples/` — small inputs, so every Quickstart command in the README
runs straight after a clone. The full sets used for the paper are too large for git and are
distributed as a release instead:

```bash
python datasets/download_data.py --status            # list the sets and which are already downloaded
python datasets/download_data.py uncond/sinddm15     # download one set
python datasets/download_data.py --all               # download every set (about 450 MB, mostly gigapixel)
```

Sets are named `<app>/<set>` and land at `datasets/<app>/<set>/` — `datasets/` is gitignored
apart from this README and the script. A file already in place is skipped, so rerunning after a
failed transfer fetches only what is missing. `small` sets are for an app's `pixel_*` presets,
`megapixel` sets for its `latent_*` preset.

| Set | Size | What it is |
|---|---:|---|
| `uncond/sinddm15` | 1.5 MB | the 15-image benchmark scored by `configs/uncond/paper_sinddm15.yaml` (Table 1) |
| `uncond/megapixel` | 32 MB | 21 inputs at ~1 Mpx: landscapes, Hubble images, two paintings, the Moon |
| `uncond/gigapixel` | 394 MB | the three inputs of the README's Gigapixel section |
| `retarget/small` | 1.5 MB | the classic retargeting test images |
| `symmetric/small` | 0.6 MB | inputs for symmetric generation, pixel presets |
| `symmetric/megapixel` | 8.2 MB | inputs for symmetric generation, latent preset |
| `tileable/small` | 0.5 MB | textures for tileable generation, pixel presets |
| `tileable/megapixel` | 6.7 MB | textures for tileable generation, latent preset |
| `structural_analogy/small` | 0.05 MB | structure (`content/`) and style (`style/`) images, pixel preset; any content pairs with any style |
| `structural_analogy/megapixel` | 3.6 MB | structure and style images, latent preset |
| `text_style/small` | 0.7 MB | inputs for text-driven style transfer |

## Credits and licences

**Most of the images below are not ours, and none are covered by this repository's MIT licence.**
Each is under its own terms, reproduced here with the attribution those terms require. Many are
the standard test images of single-image generation, taken from the codebases of
[SinGAN](https://github.com/tamarott/SinGAN), [InGAN](https://github.com/assafshocher/InGAN),
[GPNN](https://github.com/iyttor/GPNN), [GPDM](https://github.com/ariel415el/GPDM) and
[SinDDM](https://github.com/fallenshock/SinDDM); those repositories do not record the
photographers, so where a section below names a codebase, that is the source as far as it is
known, and we claim no rights. All three gigapixel sources are **NonCommercial**: they may be used
for research and teaching, but not commercially, whatever licence the surrounding code carries.

### `uncond/gigapixel`

| File | Work | By | Licence |
|---|---|---|---|
| `moon.jpg` | [Lunar Northern Near Side](https://www.flickr.com/photos/24354425@N03/16367614455/) | Stuart Rankin | [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) |
| `tokyo.jpg` | [1.2 Gigapixel Panorama of Shibuya in Tokyo, Japan](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837/) | Trevor Dobson | [CC BY-NC-ND 2.0](https://creativecommons.org/licenses/by-nc-nd/2.0/) |
| `painting.jpg` | [Impressionist Lily Ensemble](https://www.flickr.com/photos/thelastminute/53353037125/) | Duncan Rawlinson — Duncan.co | [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) |

**Modification:** `moon.jpg` is the right half of the 54582×18195 original, cropped to
27291×18195. `tokyo.jpg` and `painting.jpg` are the unmodified originals as published.

**On the NoDerivatives term.** `tokyo.jpg` is distributed here verbatim, which BY-NC-ND permits.
Generating from it produces a derivative work, which that licence does not cover — so treat any
output you make from `tokyo.jpg` as unpublishable without the photographer's permission. The other
two carry no such restriction.

### `uncond/sinddm15`

The 15-image benchmark of [SinDDM](https://github.com/fallenshock/SinDDM) (Kulikov et al.,
ICML 2023), redistributed unchanged so the Table 1 numbers can be reproduced against the same
inputs. All 15 ship in that repository's `datasets/` folder under its MIT licence; several
(`balloons`, `seascape`, `starry_night`, `mountains`, …) originate in
[SinGAN](https://github.com/tamarott/SinGAN)'s `Input/Images/` (MIT). Those licences cover the
repositories; neither documents the rights in the underlying photographs, and we claim none.

### `tileable/megapixel`

All eleven textures were generated with ChatGPT from text prompts describing the material
("weathered red brick wall", "wavy wood grain", …); no photographs. They are ours to
redistribute. `crinkled_paper.png` is the same file as `examples/tileable/crinkled_paper.png`.

### `structural_analogy/megapixel`

Ten images from the [GPDM](https://github.com/ariel415el/GPDM) repository's
`data/images/style_transfer/` (Apache-2.0 repository; its README says the images were collected
from various repositories and papers, and does not document per-image rights — we claim none),
with filenames simplified. Three portraits of political figures from that folder are deliberately
not included. `content/mr_bean.jpg` is a publicity photograph of Rowan Atkinson. `cornell.jpg`
and `thick_oil.jpg` are the same files as in `examples/structural_analogy/`.

Two further structure images:

| File | Work | Licence |
|---|---|---|
| `content/uoft_convocation_hall.jpg` | Convocation Hall, University of Toronto — [photo by Kara M on Unsplash](https://unsplash.com/photos/brown-concrete-building-during-daytime-dGsEismPga4), resized to a 1024 px short side | [Unsplash License](https://unsplash.com/license) |
| `content/mimi.jpg` | Mimi, a cat — photograph by the authors | CC0 |

### `retarget/small`

Mostly the retargeting test images of InGAN and GPNN. `broadway_tower` is Newton2's
[Wikimedia Commons photograph](https://commons.wikimedia.org/wiki/File:Broadway_tower_edit.jpg),
CC BY 2.5. `fruit` and `penguins` are the same files as in `examples/retarget/`.

### `uncond/megapixel`

All 21 are resized to about 1 Mpx.

| File | Work | Licence |
|---|---|---|
| `chrysanthemums` | Gustave Caillebotte, *Chrysanthemums in the Garden at Petit-Gennevilliers*, 1893 — [The Met](https://www.metmuseum.org/art/collection/search/671456) | CC0 (Met Open Access) |
| `vangogh` | Vincent van Gogh, *Wheat Field with Cypresses*, 1889 — [The Met](https://www.metmuseum.org/art/collection/search/436535) | CC0 (Met Open Access) |
| `whirlpool_galaxy` | [M51 and NGC 5195](https://esahubble.org/images/heic0506a/) — NASA, ESA, S. Beckwith (STScI), and The Hubble Heritage Team (STScI/AURA) | CC BY 4.0 |
| `ngc2525` | [NGC 2525](https://esahubble.org/images/heic2018b/) — ESA/Hubble & NASA, A. Riess and the SH0ES team; acknowledgement: Mahdi Zamani | CC BY 4.0 |
| `cosmic_reef` | [NGC 2014 and NGC 2020](https://esahubble.org/images/heic2007a/) — NASA, ESA, and STScI | CC BY 4.0 |
| `jupiter` | [Jupiter and Europa](https://esahubble.org/images/heic2017a/) — NASA, ESA, A. Simon (Goddard Space Flight Center), and M. H. Wong (University of California, Berkeley) and the OPAL team | CC BY 4.0 |
| `pillars_of_creation` | [Pillars of Creation](https://esahubble.org/images/heic1501a/) — NASA, ESA/Hubble and the Hubble Heritage Team | CC BY 4.0 |
| `moon_mare` | a crop of Stuart Rankin's [Lunar Northern Near Side](https://www.flickr.com/photos/24354425@N03/16367614455/), the same source as `gigapixel/moon.jpg` | CC BY-NC 2.0 |
| `alps`, `antelope_canyon`, `banff_mountains`, `dolomites`, `succulent`, `terraces` | Pixabay — katerinavulcova (9919976), TreptowerAlex (9889582), nunziog666 (10279796), andreisla (9889149), martinophuc (9949530), namden (9963155) | Pixabay Content License |
| `half_dome`, `mount_hood`, `trillium_lake`, `mount_assiniboine`, `tuolumne_meadows`, `spirit_island`, `moon_highlands` | stock photographs | — |

`alps`, `succulent` and `terraces` are the same files as in `examples/`.

### `symmetric/small` and `symmetric/megapixel`

`small/`: `forest`, `marinabaysands`, `mountains_2` are sinddm15 images; `colusseum` and `stone`
are from SinGAN's `Input/Images/`; `beach_umbrellas` is image 16 of the SIGD16 set and `sea_arch`
image 41 of Places50, the benchmark sets of SinGAN and its successors; `tapestry` is the same file
as `examples/symmetric/tapestry.png`. `megapixel/` holds six of the `uncond/megapixel` files,
credited above.

### `tileable/small`

Classic test images of the texture-synthesis literature. `bricks` is the same file as
`examples/tileable/bricks.png`; `olives` is the GPDM file also in `retarget/small`.

### `structural_analogy/small`

Both pairs are GPNN's. `s_char` and `duck_mosaic` are the same files as in
`examples/structural_analogy/`.

### `text_style/small`

All from the SinDDM paper; five (`aurora`, `forest`, `mountains_2`, `night_sky`, `seascape`) are
also in sinddm15. `bagan` is the same file as `examples/text_style/bagan.png`.

## Takedown

If you hold rights in any image here and would rather it were not redistributed, open an issue and
it will be removed from the release.
