"""
bench/prepare_realset.py — Download ~300 real photos and generate app-style duplicates.

Source: vikhyatk/revisitop (ROxford5k benchmark subset, VGG Oxford / Philbin et al. / Radenović et al.)
License: Creative Commons Attribution (Flickr CC-BY) & Oxford Non-Commercial Research License.
Permits research/evaluation use.
"""

from __future__ import annotations

import csv
from pathlib import Path
import random
import shutil
import sys

from datasets import load_dataset
from PIL import Image


def prepare_realset(
    target_count: int = 300,
    duplicate_count: int = 40,
    seed: int = 42,
    base_dir: str | Path = "~/dedupe_test/realset",
) -> None:
    base_path = Path(base_dir).expanduser().resolve()
    originals_dir = base_path / "originals"
    photos_dir = base_path / "photos"

    originals_dir.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("  Downloading Real Photographs (ROxford5k Subset)")
    print("==================================================")
    print(f"Dataset : vikhyatk/revisitop (roxford5k)")
    print(f"Target  : {target_count} real photographs")
    print(f"Storage : {originals_dir}")

    # Shards containing the images in parquet format
    shards = [
        "roxford5k/imlist-00000-of-00020.parquet",
        "roxford5k/imlist-00001-of-00020.parquet",
    ]

    ds = load_dataset(
        "vikhyatk/revisitop",
        data_files={"imlist": shards},
        split="imlist",
        streaming=True,
    )

    saved_originals: list[str] = []
    total_bytes = 0

    for item in ds:
        fname = item["filename"]
        if not fname.endswith(".jpg"):
            fname = f"{fname}.jpg"

        orig_dest = originals_dir / fname
        photo_dest = photos_dir / fname

        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save to originals
        img.save(orig_dest, format="JPEG", quality=95)
        # Also copy to photos folder
        shutil.copy2(orig_dest, photo_dest)

        file_sz = orig_dest.stat().st_size
        total_bytes += file_sz
        saved_originals.append(fname)

        if len(saved_originals) % 50 == 0 or len(saved_originals) == target_count:
            print(f"  Downloaded: {len(saved_originals)}/{target_count} images ({total_bytes / (1024*1024):.2f} MB)...")

        if len(saved_originals) >= target_count:
            break

    print("--------------------------------------------------")
    print(f"Originals ready: {len(saved_originals)} images, {total_bytes / (1024*1024):.2f} MB")
    print("==================================================")

    # Step 2: Make app-style duplicates from 40 random originals
    print("\n==================================================")
    print(f"  Generating App-Style Duplicates ({duplicate_count} originals, seed={seed})")
    print("==================================================")

    rng = random.Random(seed)
    sorted_origs = sorted(saved_originals)
    sampled_origs = sorted(rng.sample(sorted_origs, duplicate_count))

    gt_rows: list[dict[str, str]] = []

    # Record all originals in ground truth
    for orig in sorted_origs:
        gt_rows.append({
            "file": orig,
            "original": orig,
            "transformation": "original",
        })

    for orig_fname in sampled_origs:
        orig_path = originals_dir / orig_fname
        stem = orig_path.stem
        orig_img = Image.open(orig_path)
        w, h = orig_img.size

        # (a) resize long edge to 1600 and JPEG q~75 with EXIF stripped (WhatsApp-like)
        wa_fname = f"{stem}_wa_1600_q75.jpg"
        wa_path = photos_dir / wa_fname
        if max(w, h) > 1600:
            scale = 1600.0 / max(w, h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_wa = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img_wa = orig_img.copy()
        if img_wa.mode != "RGB":
            img_wa = img_wa.convert("RGB")
        # Save without EXIF
        img_wa.save(wa_path, format="JPEG", quality=75)
        gt_rows.append({
            "file": wa_fname,
            "original": orig_fname,
            "transformation": "whatsapp_1600_q75",
        })

        # (b) resize to 1080 q~70
        res_fname = f"{stem}_res_1080_q70.jpg"
        res_path = photos_dir / res_fname
        if max(w, h) > 1080:
            scale = 1080.0 / max(w, h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_res = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img_res = orig_img.copy()
        if img_res.mode != "RGB":
            img_res = img_res.convert("RGB")
        img_res.save(res_path, format="JPEG", quality=70)
        gt_rows.append({
            "file": res_fname,
            "original": orig_fname,
            "transformation": "resize_1080_q70",
        })

        # (c) re-encode a screenshot-like PNG at 1x
        png_fname = f"{stem}_screenshot.png"
        png_path = photos_dir / png_fname
        orig_img.save(png_path, format="PNG")
        gt_rows.append({
            "file": png_fname,
            "original": orig_fname,
            "transformation": "screenshot_png_1x",
        })

        # (d) an exact byte copy
        copy_fname = f"{stem}_copy.jpg"
        copy_path = photos_dir / copy_fname
        shutil.copyfile(orig_path, copy_path)
        gt_rows.append({
            "file": copy_fname,
            "original": orig_fname,
            "transformation": "exact_byte_copy",
        })

    # Write ground_truth.csv
    csv_paths = [
        photos_dir / "ground_truth.csv",
        base_path / "ground_truth.csv",
    ]
    for cp in csv_paths:
        with open(cp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "original", "transformation"])
            writer.writeheader()
            writer.writerows(gt_rows)

    total_photos = len(list(p for p in photos_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    total_photo_bytes = sum(p.stat().st_size for p in photos_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    print("--------------------------------------------------")
    print(f"Transformations per sampled original: 4")
    print(f"Sampled originals   : {len(sampled_origs)}")
    print(f"Generated duplicates: {len(sampled_origs) * 4}")
    print(f"Total photos in set : {total_photos} ({total_photo_bytes / (1024*1024):.2f} MB)")
    print(f"Ground truth written: {csv_paths[0]} ({len(gt_rows)} entries)")
    print("==================================================")


if __name__ == "__main__":
    prepare_realset()
