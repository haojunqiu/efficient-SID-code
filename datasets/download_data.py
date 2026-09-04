"""Fetch the datasets that the shipped configs and the paper's Table 1 use.

The repo commits only ``examples/`` -- small inputs, so every Quickstart command runs straight
after a clone. The full sets are too large for git; they live on a GitHub release and land
here in ``datasets/``, which is gitignored apart from this script.

    uncond/sinddm15                15 images    1.5 MB   scored by configs/uncond/paper_sinddm15.yaml
    uncond/megapixel               21 images     32 MB   the ~1 Mpx unconditional inputs
    uncond/gigapixel                3 images    394 MB   the moon / tokyo / painting figures
    retarget/small                 14 images    1.5 MB   the classic retargeting test images
    symmetric/small                10 images    0.6 MB   symmetric inputs, pixel preset
    symmetric/megapixel             6 images    8.2 MB   symmetric inputs, latent preset
    tileable/small                  8 textures  0.5 MB   tileable inputs, pixel preset
    tileable/megapixel             11 textures  6.7 MB   tileable inputs, latent preset
    structural_analogy/small        4 images   0.05 MB   structure/style pairs, pixel preset
    structural_analogy/megapixel   12 images    3.6 MB   structure/style pairs, latent preset
    text_style/small                8 images    0.7 MB   text-driven style inputs

Sets are named <app>/<set>, matching where they land. A file already in place is skipped, so
rerunning after a failed transfer fetches only what is missing. The images are third-party
works under their own licences -- see datasets/README.md.

Usage (from the repo root):

    python datasets/download_data.py --status            # list the sets and which are already downloaded
    python datasets/download_data.py uncond/sinddm15     # download one set
    python datasets/download_data.py --all               # download every set

Standard library only: this has to run before the environment is installed.
"""
import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/haojunqiu/efficient-SID-code/releases/download/data-v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

#: set -> (size, {release asset: where it lands, relative to the repo root}).
#: A .zip is unpacked into its directory; any other asset is saved as the file named.
SETS = {
    "uncond/sinddm15": ("1.5 MB", {
        "uncond-sinddm15.zip": "datasets/uncond/sinddm15/",
    }),
    "uncond/megapixel": ("32 MB", {
        "uncond-megapixel.zip": "datasets/uncond/megapixel/",
    }),
    "uncond/gigapixel": ("394 MB", {
        "uncond-gigapixel-moon.jpg": "datasets/uncond/gigapixel/moon.jpg",
        "uncond-gigapixel-tokyo.jpg": "datasets/uncond/gigapixel/tokyo.jpg",
        "uncond-gigapixel-painting.jpg": "datasets/uncond/gigapixel/painting.jpg",
    }),
    "retarget/small": ("1.5 MB", {
        "retarget-small.zip": "datasets/retarget/small/",
    }),
    "symmetric/small": ("0.6 MB", {
        "symmetric-small.zip": "datasets/symmetric/small/",
    }),
    "symmetric/megapixel": ("8.2 MB", {
        "symmetric-megapixel.zip": "datasets/symmetric/megapixel/",
    }),
    "tileable/small": ("0.5 MB", {
        "tileable-small.zip": "datasets/tileable/small/",
    }),
    "tileable/megapixel": ("6.7 MB", {
        "tileable-megapixel.zip": "datasets/tileable/megapixel/",
    }),
    "structural_analogy/small": ("0.05 MB", {
        "structural_analogy-small.zip": "datasets/structural_analogy/small/",
    }),
    "structural_analogy/megapixel": ("3.6 MB", {
        "structural_analogy-megapixel.zip": "datasets/structural_analogy/megapixel/",
    }),
    "text_style/small": ("0.7 MB", {
        "text_style-small.zip": "datasets/text_style/small/",
    }),
}


def in_place(asset, target):
    """A zip counts once its directory has anything in it; a file once it exists."""
    if asset.endswith(".zip"):
        return target.is_dir() and any(target.iterdir())
    return target.exists()


def download(asset, dest):
    """Save one release asset to dest, or leave nothing behind."""
    request = urllib.request.Request(f"{RELEASE}/{asset}",
                                     headers={"User-Agent": "efficient-sid"})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise SystemExit(f"  {asset}: not on the release at {RELEASE}\n"
                             "  Is the data release published yet?") from None
        raise
    expected = response.headers.get("Content-Length")
    print(f"  {asset}" + (f"  ({int(expected) / 1e6:.1f} MB)" if expected else ""))

    part = dest.with_name(dest.name + ".part")
    with response, part.open("wb") as out:
        shutil.copyfileobj(response, out)
    written = part.stat().st_size
    if expected and written != int(expected):
        part.unlink()
        raise SystemExit(f"  {asset}: transfer cut short at {written} of {expected} bytes "
                         "-- rerun to retry")
    part.replace(dest)


def fetch(name):
    _, assets = SETS[name]
    pending = {a: REPO_ROOT / p for a, p in assets.items() if not in_place(a, REPO_ROOT / p)}
    if not pending:
        print(f"{name}: already in place")
        return
    print(f"{name}:")
    for asset, target in pending.items():
        if asset.endswith(".zip"):
            target.mkdir(parents=True, exist_ok=True)
            archive = target / asset
            download(asset, archive)
            with zipfile.ZipFile(archive) as unpack:
                unpack.extractall(target)
            archive.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            download(asset, target)


def status():
    for name, (size, assets) in SETS.items():
        missing = sum(not in_place(a, REPO_ROOT / p) for a, p in assets.items())
        state = "in place" if not missing else f"{missing} of {len(assets)} missing"
        print(f"  {name:<30} {size:>7}   {state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sets", nargs="*", metavar="SET",
                        help=f"one or more of: {', '.join(SETS)}")
    parser.add_argument("--all", action="store_true", help="fetch every set")
    parser.add_argument("--status", action="store_true",
                        help="list the sets and whether each is in place; fetch nothing")
    args = parser.parse_args()

    unknown = [s for s in args.sets if s not in SETS]
    if unknown:
        parser.error(f"no such set: {', '.join(unknown)}; choose from {', '.join(SETS)}")
    if args.status:
        status()
        return
    if not args.sets and not args.all:
        parser.print_help()
        return
    for name in (SETS if args.all else args.sets):
        fetch(name)
    print("\nCredits and licences: datasets/README.md")


if __name__ == "__main__":
    main()
