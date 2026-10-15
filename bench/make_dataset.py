"""
bench/make_dataset.py — Benchmark dataset generator for Argus.

Given a folder of original images (data/originals/ by default), produces
data/benchmark/ containing the originals plus automatically generated variants
of every type the pipeline must detect, plus a ground_truth.csv manifest.

Transformation types generated
───────────────────────────────
  original          — the source file copied verbatim
  exact_copy        — byte-for-byte identical copy (different filename)
  resize            — thumbnail at several edge lengths
  jpeg_recompress   — saved as JPEG at several quality levels
  format_png        — losslessly converted to PNG
  format_webp       — converted to WEBP
  crop              — slight crop (2% / 5% from each edge)
  brightness        — ±15% brightness
  contrast          — ±15% contrast
  rotate90          — rotated 90° counter-clockwise

Design constraints
──────────────────
  • One image in memory at a time — no bulk loading.
  • All transform parameters live in config.toml [benchmark].
  • HEIC supported via pillow-heif.
  • RAW files that Pillow cannot decode are recorded as originals only
    (no variants); a warning is printed.
  • Exact copies use shutil.copy2 to guarantee identical bytes.
  • Variant filenames: {original_stem}__{tag}.{ext}
    If two originals share the same stem, the extension is appended to
    the stem to avoid collisions: {stem}_{ext_no_dot}__{tag}.{ext}

ground_truth.csv columns
─────────────────────────
  filename            — path relative to output_dir
  original_filename   — basename of the source image
  transformation      — transformation tag (see list above)
  parameters          — human-readable parameter string (e.g. "quality=85")

Usage
─────
  python bench/make_dataset.py
  python bench/make_dataset.py --originals data/originals --output data/benchmark
  python bench/make_dataset.py --limit 10   # process first 10 originals only
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pillow_heif
from PIL import Image, ImageEnhance

# Register HEIC/HEIF support globally.
pillow_heif.register_heif_opener()

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class VariantRecord:
    """One row in ground_truth.csv."""
    filename: str           # path relative to output_dir
    original_filename: str  # basename of the source file
    transformation: str     # tag string
    parameters: str         # human-readable params, empty string for originals


@dataclass
class DatasetStats:
    originals_processed: int = 0
    originals_skipped: int = 0    # could not be opened by Pillow
    variants_created: int = 0
    total_files: int = 0


# ---------------------------------------------------------------------------
# Extensions Pillow can open (via base + pillow-heif plugin)
# RAW formats (.arw etc.) are not supported without rawpy; they are skipped.
# ---------------------------------------------------------------------------

_OPENABLE_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".bmp", ".tiff", ".tif", ".heic", ".heif",
})

_ALL_EXTS: frozenset[str] = _OPENABLE_EXTS | frozenset({
    ".arw", ".cr2", ".cr3", ".nef", ".orf",
    ".rw2", ".dng", ".raf", ".pef",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_originals(originals_dir: Path) -> Iterator[Path]:
    """Yield image files from originals_dir, sorted for reproducibility."""
    for p in sorted(originals_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in _ALL_EXTS:
            yield p


def _unique_stem(path: Path, seen_stems: set[str]) -> str:
    """
    Return a variant-name stem that is unique within *seen_stems*.
    Normally this is just path.stem.  If that stem is already taken
    (two originals with the same name but different extensions), append
    the extension without the dot: 'photo_jpg', 'photo_png'.
    """
    stem = path.stem
    if stem not in seen_stems:
        seen_stems.add(stem)
        return stem
    stem_with_ext = f"{stem}_{path.suffix.lstrip('.').lower()}"
    seen_stems.add(stem_with_ext)
    return stem_with_ext


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert palette / RGBA / other modes to RGB for JPEG/WEBP saving."""
    if img.mode in ("RGB",):
        return img
    if img.mode == "RGBA":
        # Composite onto white background before dropping alpha.
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB")


def _open_rgb(src: Path) -> Image.Image | None:
    """
    Open *src* and return it as an RGB Image.
    Returns None if Pillow cannot decode the file (e.g. RAW formats).
    """
    if src.suffix.lower() not in _OPENABLE_EXTS:
        return None
    try:
        img = Image.open(src)
        img.load()          # force decode now while file is open
        return _to_rgb(img)
    except Exception as exc:
        warnings.warn(f"Cannot open {src.name}: {exc}", stacklevel=2)
        return None


def _save_jpeg(img: Image.Image, dest: Path, quality: int) -> None:
    img.save(dest, format="JPEG", quality=quality, optimize=True)


# ---------------------------------------------------------------------------
# Per-image transform functions
# Each returns a list[VariantRecord] and writes files as a side-effect.
# All functions receive an already-open RGB image — no file I/O for reads.
# ---------------------------------------------------------------------------

def _make_exact_copy(
    src: Path,
    out_dir: Path,
    stem: str,
) -> VariantRecord:
    """Byte-for-byte copy using shutil.copy2 (preserves timestamps too)."""
    dest = out_dir / f"{stem}__exact{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return VariantRecord(
        filename=dest.name,
        original_filename=src.name,
        transformation="exact_copy",
        parameters="",
    )


def _make_resizes(
    img: Image.Image,
    src: Path,
    out_dir: Path,
    stem: str,
    sizes: list[int],
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    orig_max = max(img.size)
    for size in sizes:
        if size >= orig_max:
            # Don't upscale — that is not a near-duplicate scenario.
            continue
        resized = img.copy()
        resized.thumbnail((size, size), Image.LANCZOS)
        dest = out_dir / f"{stem}__resize_{size}.jpg"
        _save_jpeg(resized, dest, quality=95)
        records.append(VariantRecord(
            filename=dest.name,
            original_filename=src.name,
            transformation="resize",
            parameters=f"size={size}",
        ))
        resized.close()
    return records


def _make_jpeg_recompressed(
    img: Image.Image,
    src: Path,
    out_dir: Path,
    stem: str,
    qualities: list[int],
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    for quality in qualities:
        dest = out_dir / f"{stem}__jpeg_q{quality}.jpg"
        _save_jpeg(img, dest, quality=quality)
        records.append(VariantRecord(
            filename=dest.name,
            original_filename=src.name,
            transformation="jpeg_recompress",
            parameters=f"quality={quality}",
        ))
    return records


def _make_format_converts(
    img: Image.Image,
    src: Path,
    out_dir: Path,
    stem: str,
    formats: list[str],
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    ext_map = {"PNG": ".png", "WEBP": ".webp"}
    for fmt in formats:
        ext = ext_map.get(fmt.upper(), f".{fmt.lower()}")
        dest = out_dir / f"{stem}__fmt_{fmt.lower()}{ext}"
        save_kwargs: dict = {"format": fmt}
        if fmt.upper() == "WEBP":
            save_kwargs["quality"] = 90
        # PNG supports RGBA; we already have RGB so this is fine.
        img.save(dest, **save_kwargs)
        records.append(VariantRecord(
            filename=dest.name,
            original_filename=src.name,
            transformation=f"format_{fmt.lower()}",
            parameters=f"format={fmt}",
        ))
    return records


def _make_crops(
    img: Image.Image,
    src: Path,
    out_dir: Path,
    stem: str,
    fractions: list[float],
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    w, h = img.size
    for frac in fractions:
        dx = max(1, int(w * frac))
        dy = max(1, int(h * frac))
        box = (dx, dy, w - dx, h - dy)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue  # image too small for this fraction
        cropped = img.crop(box)
        tag = str(int(frac * 100)).zfill(2)
        dest = out_dir / f"{stem}__crop_{tag}pct.jpg"
        _save_jpeg(cropped, dest, quality=95)
        records.append(VariantRecord(
            filename=dest.name,
            original_filename=src.name,
            transformation="crop",
            parameters=f"fraction={frac}",
        ))
        cropped.close()
    return records


def _make_brightness(
    img: Image.Image,
    src: Path,
    out_dir: Path,
    stem: str,
    factors: list[float],
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    enhancer = ImageEnhance.Brightness(img)
    for factor in factors:
        adjusted = enhancer.enhance(factor)
        tag = str(int(factor * 100)).zfill(3)
        dest = out_dir / f"{stem}__bright_{tag}.jpg"
        _save_jpeg(adjusted, dest, quality=95)
        records.append(VariantRecord(
            filename=dest.name,
            original_filename=src.name,
            transformation="brightness",
            parameters=f"factor={factor}",
        ))
        adjusted.close()
    return records


def _make_contrast(
    img: Image.Image,
    src: Path,
    out_dir: Path,
    stem: str,
    factors: list[float],
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    enhancer = ImageEnhance.Contrast(img)
    for factor in factors:
        adjusted = enhancer.enhance(factor)
        tag = str(int(factor * 100)).zfill(3)
        dest = out_dir / f"{stem}__contrast_{tag}.jpg"
        _save_jpeg(adjusted, dest, quality=95)
        records.append(VariantRecord(
            filename=dest.name,
            original_filename=src.name,
            transformation="contrast",
            parameters=f"factor={factor}",
        ))
        adjusted.close()
    return records


def _make_rotations(
    img: Image.Image,
    src: Path,
    out_dir: Path,
    stem: str,
    angles: list[int],
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    for angle in angles:
        # expand=True keeps the full image visible after rotation.
        rotated = img.rotate(angle, expand=True)
        dest = out_dir / f"{stem}__rotate{angle}.jpg"
        _save_jpeg(rotated, dest, quality=95)
        records.append(VariantRecord(
            filename=dest.name,
            original_filename=src.name,
            transformation=f"rotate{angle}",
            parameters=f"angle={angle}",
        ))
        rotated.close()
    return records


# ---------------------------------------------------------------------------
# Per-image orchestration
# ---------------------------------------------------------------------------

def _process_one(
    src: Path,
    out_dir: Path,
    stem: str,
    cfg_bench: dict,
) -> list[VariantRecord]:
    """
    Process a single original image.  Opens it once, runs all transforms,
    closes it.  Returns all VariantRecords including the original row.
    """
    records: list[VariantRecord] = []

    # Always record the original file itself.
    original_dest = out_dir / src.name
    if not original_dest.exists():
        shutil.copy2(src, original_dest)
    records.append(VariantRecord(
        filename=src.name,
        original_filename=src.name,
        transformation="original",
        parameters="",
    ))

    # Open the image for variant generation.
    img = _open_rgb(src)
    if img is None:
        # RAW or unreadable — record original only.
        warnings.warn(
            f"Skipping variants for {src.name} (cannot be opened by Pillow).",
            stacklevel=2,
        )
        return records

    try:
        records.append(_make_exact_copy(src, out_dir, stem))

        records.extend(_make_resizes(
            img, src, out_dir, stem,
            sizes=cfg_bench.get("resize_sizes", [128, 256, 512, 1024]),
        ))
        records.extend(_make_jpeg_recompressed(
            img, src, out_dir, stem,
            qualities=cfg_bench.get("jpeg_qualities", [95, 85, 75, 60]),
        ))
        records.extend(_make_format_converts(
            img, src, out_dir, stem,
            formats=cfg_bench.get("format_targets", ["PNG", "WEBP"]),
        ))
        records.extend(_make_crops(
            img, src, out_dir, stem,
            fractions=cfg_bench.get("crop_fractions", [0.02, 0.05]),
        ))
        records.extend(_make_brightness(
            img, src, out_dir, stem,
            factors=cfg_bench.get("brightness_factors", [0.85, 1.15]),
        ))
        records.extend(_make_contrast(
            img, src, out_dir, stem,
            factors=cfg_bench.get("contrast_factors", [0.85, 1.15]),
        ))
        records.extend(_make_rotations(
            img, src, out_dir, stem,
            angles=cfg_bench.get("rotate_angles", [90]),
        ))
    finally:
        img.close()

    return records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = ["filename", "original_filename", "transformation", "parameters"]


def build_dataset(
    originals_dir: Path,
    output_dir: Path,
    cfg_bench: dict,
    limit: int | None = None,
    verbose: bool = True,
) -> DatasetStats:
    """
    Build the full benchmark dataset.

    Parameters
    ----------
    originals_dir : directory containing source images.
    output_dir    : destination directory (created if absent).
    cfg_bench     : the [benchmark] section of config.toml.
    limit         : if given, process only the first *limit* originals.
    verbose       : print progress to stdout.

    Returns DatasetStats with counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = DatasetStats()
    seen_stems: set[str] = set()

    gt_path = output_dir / cfg_bench.get("ground_truth_filename", "ground_truth.csv")
    with open(gt_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for src in _iter_originals(originals_dir):
            if limit is not None and stats.originals_processed >= limit:
                break

            stem = _unique_stem(src, seen_stems)

            if verbose:
                print(f"  [{stats.originals_processed + 1}] {src.name}", end=" ... ", flush=True)

            records = _process_one(src, output_dir, stem, cfg_bench)

            for rec in records:
                writer.writerow({
                    "filename": rec.filename,
                    "original_filename": rec.original_filename,
                    "transformation": rec.transformation,
                    "parameters": rec.parameters,
                })
            csv_file.flush()  # crash-safe: each image's rows land immediately

            variants = len(records) - 1  # exclude the original row
            stats.originals_processed += 1
            stats.variants_created += variants
            stats.total_files += len(records)

            if verbose:
                print(f"{variants} variants")

    if verbose:
        print(
            f"\nDone. {stats.originals_processed} originals → "
            f"{stats.variants_created} variants → "
            f"{stats.total_files} total files."
        )
        print(f"Ground truth: {gt_path}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a benchmark dataset for Argus duplicate detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--originals",
        type=Path,
        default=None,
        help="Folder of original images (overrides config.toml benchmark.originals_dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (overrides config.toml benchmark.output_dir).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: repo-root config.toml).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N originals (useful for quick tests).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Import here to avoid circular deps if this module is imported in tests.
    from core.config import load_config, DEFAULT_CONFIG_PATH

    config_path = args.config or DEFAULT_CONFIG_PATH
    cfg = load_config(config_path)
    cfg_bench: dict = cfg.get("benchmark", {})

    originals_dir = args.originals or Path(cfg_bench.get("originals_dir", "data/originals"))
    output_dir = args.output or Path(cfg_bench.get("output_dir", "data/benchmark"))

    if not originals_dir.exists():
        print(f"ERROR: originals directory not found: {originals_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Originals : {originals_dir.resolve()}")
    print(f"Output    : {output_dir.resolve()}")
    if args.limit:
        print(f"Limit     : {args.limit} images")
    print()

    build_dataset(
        originals_dir=originals_dir,
        output_dir=output_dir,
        cfg_bench=cfg_bench,
        limit=args.limit,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
