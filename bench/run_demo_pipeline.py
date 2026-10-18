"""
bench/run_demo_pipeline.py — Create ~/dedupe_test/demo and run the full pipeline.

Sets up 6 benchmark originals with:
- original
- exact copy
- q85 JPEG
- q60 JPEG
- q30 JPEG
- 50% scale JPEG

Runs:
- Stage 0: scan & MD5 exact matching
- Stage 1: perceptual hash near-duplicates
- Stage 2: CLIP embedding & FAISS clustering
- Stage 3: SIFT keypoint verification
- Stage 4 prep: Quality scoring & proxy alignment
- Stage 4: Keeper selection across conservative, balanced, and aggressive modes
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

from PIL import Image

from core.cli import main as cli_main
from core.cluster import run_stage2_clustering
from core.config import load_config
from core.embeddings import run_embeddings
from core.keeper import choose_keeper, DEFAULT_KEEPER_MODES, ensure_keeper_columns
from core.quality import run_quality_pipeline
from core.stage0 import run_stage0
from core.stage1 import run_stage1
from core.verify import run_stage3_verification
from db.connection import init_db
from sources.local_folder import LocalFolderSource


def setup_demo_folder(demo_dir: Path) -> list[str]:
    demo_dir.mkdir(parents=True, exist_ok=True)
    for p in list(demo_dir.iterdir()):
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    originals = [
        "picsum_0001.jpg",
        "picsum_0003.jpg",
        "picsum_0004.jpg",
        "picsum_0005.jpg",
        "picsum_0006.jpg",
        "picsum_0011.jpg",
    ]

    created_files = []
    for orig_name in originals:
        orig_src = Path("data/originals") / orig_name
        stem = orig_src.stem

        # 1. Original
        orig_dest = demo_dir / f"{stem}.jpg"
        shutil.copy2(orig_src, orig_dest)
        created_files.append(orig_dest.name)

        # 2. Exact copy
        exact_dest = demo_dir / f"{stem}__exact_copy.jpg"
        shutil.copy2(orig_src, exact_dest)
        created_files.append(exact_dest.name)

        # Load image for variants
        with Image.open(orig_src) as im:
            im_rgb = im.convert("RGB")

            # 3. q85 JPEG
            q85_dest = demo_dir / f"{stem}__jpeg_q85.jpg"
            im_rgb.save(q85_dest, "JPEG", quality=85, optimize=True)
            created_files.append(q85_dest.name)

            # 4. q60 JPEG
            q60_dest = demo_dir / f"{stem}__jpeg_q60.jpg"
            im_rgb.save(q60_dest, "JPEG", quality=60, optimize=True)
            created_files.append(q60_dest.name)

            # 5. q30 JPEG
            q30_dest = demo_dir / f"{stem}__jpeg_q30.jpg"
            im_rgb.save(q30_dest, "JPEG", quality=30, optimize=True)
            created_files.append(q30_dest.name)

            # 6. 50% scale JPEG
            w, h = im_rgb.size
            scale50_dest = demo_dir / f"{stem}__scale_50pct.jpg"
            im_scaled = im_rgb.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
            im_scaled.save(scale50_dest, "JPEG", quality=90)
            created_files.append(scale50_dest.name)

    return created_files


def run_demo():
    demo_dir = Path.home() / "dedupe_test" / "demo"
    print(f"Setting up demo images in: {demo_dir}")
    files = setup_demo_folder(demo_dir)
    print(f"Created {len(files)} demo files across 6 original scenes.\n")

    cfg = load_config()
    db_path = demo_dir / "demo.db"
    if db_path.exists():
        db_path.unlink()

    conn = init_db(str(db_path))
    source = LocalFolderSource(demo_dir)

    print("=" * 80)
    print("  ARGUS FULL PIPELINE EXECUTION ON DEMO DATASET")
    print("=" * 80)

    # 1. Stage 0
    print("[1/5] Running Stage 0: File discovery, MD5 hashing, exact clustering...")
    t0 = time.perf_counter()
    res_s0 = run_stage0(source, conn, cfg)
    print(f"      Files hashed: {res_s0.files_hashed}, Exact clusters: {res_s0.clusters_created} ({res_s0.files_clustered} files) [{time.perf_counter()-t0:.2f}s]")

    # 2. Stage 1
    print("[2/5] Running Stage 1: Perceptual hashing (pHash)...")
    t0 = time.perf_counter()
    res_s1 = run_stage1(source, conn, cfg)
    print(f"      pHashes computed: {res_s1.hashes_computed}, Clusters formed: {res_s1.clusters_created} [{time.perf_counter()-t0:.2f}s]")

    # 3. Stage 2
    print("[3/5] Running Stage 2: CLIP embeddings and FAISS clustering...")
    t0 = time.perf_counter()
    res_emb = run_embeddings(source, conn, cfg)
    faiss_path = demo_dir / "demo.faiss"
    res_s2 = run_stage2_clustering(conn, cfg, index_path=faiss_path)
    print(f"      Embedded: {res_emb.images_embedded}, S2 clusters created: {res_s2.clusters_created}, Merged: {res_s2.clusters_merged} [{time.perf_counter()-t0:.2f}s]")

    # 4. Stage 3
    print("[4/5] Running Stage 3: SIFT geometric verification & RANSAC...")
    t0 = time.perf_counter()
    res_s3 = run_stage3_verification(conn, cfg, source=source)
    print(f"      Pairs evaluated: {res_s3.pairs_evaluated}, Verified: {res_s3.pairs_verified}, Split: {res_s3.members_split} [{time.perf_counter()-t0:.2f}s]")

    # 5. Stage 4 Quality Scoring
    print("[5/5] Running Stage 4 Quality Scoring & 1024px Proxy Alignment...")
    t0 = time.perf_counter()
    res_q = run_quality_pipeline(conn, cfg, source=source)
    print(f"      Clusters evaluated: {res_q.clusters_evaluated}, Members aligned: {res_q.members_aligned}, Unalignable: {res_q.members_unalignable} [{time.perf_counter()-t0:.2f}s]\n")

    # Fetch clusters
    clusters = conn.execute("SELECT id, representative_id FROM clusters ORDER BY id").fetchall()

    for mode in ["conservative", "balanced", "aggressive"]:
        print("\n" + "=" * 80)
        print(f"  STAGE 4 KEEPER RESULTS — MODE: {mode.upper()}")
        print("=" * 80)
        grand_total_savings = 0
        total_keepers = 0
        total_delete_safe = 0
        total_review = 0

        for c in clusters:
            cid = c["id"]
            members = conn.execute(
                """SELECT id, file_path, file_size, width, height, format, md5, phash,
                          status, cluster_id, is_keeper, is_low_confidence,
                          ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, quality_status
                   FROM images WHERE cluster_id = ? ORDER BY id""",
                (cid,)
            ).fetchall()
            if not members:
                continue

            res = choose_keeper(members, mode=mode, config=cfg)
            id_to_m = {m["id"]: m for m in members}
            k_m = id_to_m[res["keeper_id"]]
            k_name = Path(k_m["file_path"]).name

            total_keepers += 1
            total_delete_safe += len(res["delete_safe_ids"])
            total_review += len(res["review_ids"])
            grand_total_savings += res["total_savings_bytes"]

            print(f"Cluster {cid} ({len(members)} members):")
            print(f"  Keeper     : {k_name} ({k_m['file_size']:,} B, {k_m['width']}x{k_m['height']}, quality={res['quality_retained'][res['keeper_id']]}%)")
            print(f"               Reason: {res['reasons'][res['keeper_id']]}")

            print(f"  Delete-Safe: ({len(res['delete_safe_ids'])} files, {res['total_savings_bytes']:,} bytes saved)")
            for ds_id in res["delete_safe_ids"]:
                ds_m = id_to_m[ds_id]
                ds_name = Path(ds_m["file_path"]).name
                print(f"    - {ds_name:<26} : {ds_m['file_size']:,} B (Quality: {res['quality_retained'][ds_id]:>5.1f}%) — {res['reasons'][ds_id]}")

            print(f"  Review     : ({len(res['review_ids'])} files)")
            if res["review_ids"]:
                for rv_id in res["review_ids"]:
                    rv_m = id_to_m[rv_id]
                    rv_name = Path(rv_m["file_path"]).name
                    print(f"    - {rv_name:<26} : {rv_m['file_size']:,} B — {res['reasons'][rv_id]}")
            else:
                print("    (none)")

            print(f"  Cluster Savings: {res['total_savings_bytes']:,} bytes ({res['total_savings_bytes'] / 1024:.1f} KB)\n")

        print("-" * 80)
        print(f"SUMMARY FOR {mode.upper()} MODE:")
        print(f"  Total Clusters Evaluated   : {len(clusters)}")
        print(f"  Recommended Keepers        : {total_keepers}")
        print(f"  Files Marked Delete-Safe   : {total_delete_safe}")
        print(f"  Files Flagged for Review   : {total_review}")
        print(f"  Grand Total Space Reclaimed: {grand_total_savings:,} bytes ({grand_total_savings / (1024*1024):.2f} MB)")
        print("=" * 80)


if __name__ == "__main__":
    run_demo()
