"""
bench/download_originals.py — Download 200 CC0 images from Lorem Picsum.

Images come from https://picsum.photos (sourced from Unsplash, CC0 licensed).
Each image is downloaded at 1024×768 (landscape) using a stable numeric ID
so the dataset is fully reproducible.

Usage:
    python bench/download_originals.py
    python bench/download_originals.py --target 50   # fewer images
    python bench/download_originals.py --size 800    # different resolution
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Picsum IDs are stable integers; we try IDs 1-300 to land 200 successes
# (a small number of IDs are retired/missing and return 404).
_ID_POOL = list(range(1, 301))

_DEFAULT_TARGET = 200
_DEFAULT_SIZE = "1024/768"   # width/height
_DEFAULT_OUT = Path("data/originals")
_DELAY = 0.05                # seconds between requests (be polite)


def download(
    target: int = _DEFAULT_TARGET,
    size: str = _DEFAULT_SIZE,
    out_dir: Path = _DEFAULT_OUT,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for img_id in _ID_POOL:
        if downloaded >= target:
            break

        dest = out_dir / f"picsum_{img_id:04d}.jpg"
        if dest.exists():
            downloaded += 1
            print(f"  [{downloaded}/{target}] picsum_{img_id:04d}.jpg  (already exists)")
            continue

        url = f"https://picsum.photos/id/{img_id}/{size}"
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            size_kb = dest.stat().st_size // 1024
            print(f"  [{downloaded}/{target}] picsum_{img_id:04d}.jpg  ({size_kb} KB)")
            time.sleep(_DELAY)
        except urllib.error.HTTPError as e:
            skipped += 1
            print(f"  skip id={img_id}: HTTP {e.code}", file=sys.stderr)
        except Exception as e:
            skipped += 1
            print(f"  skip id={img_id}: {e}", file=sys.stderr)

    print(f"\nDone. {downloaded} images downloaded to {out_dir}  ({skipped} IDs skipped)")
    return downloaded


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CC0 benchmark images from Lorem Picsum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=int, default=_DEFAULT_TARGET,
                   help="Number of images to download.")
    p.add_argument("--size", default=_DEFAULT_SIZE,
                   help="Resolution as WIDTH/HEIGHT (Picsum format).")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                   help="Output directory.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    n = download(target=args.target, size=args.size, out_dir=args.out)
    if n == 0:
        sys.exit(1)
