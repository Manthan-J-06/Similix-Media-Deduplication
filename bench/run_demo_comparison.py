"""
bench/run_demo_comparison.py — Rerun demo twice (normal names vs random strings) and compare references and keepers.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time

from core.cluster import run_stage2_clustering
from core.config import load_config
from core.embeddings import run_embeddings
from core.keeper import choose_keeper
from core.quality import run_quality_pipeline
from core.stage0 import run_stage0
from core.stage1 import run_stage1
from core.verify import run_stage3_verification
from db.connection import init_db
from sources.local_folder import LocalFolderSource
from bench.run_demo_pipeline import setup_demo_folder


def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def execute_pipeline(demo_dir: Path) -> dict:
    cfg = load_config()
    db_path = demo_dir / "demo.db"
    if db_path.exists():
        db_path.unlink()
    faiss_path = demo_dir / "demo.faiss"
    if faiss_path.exists():
        faiss_path.unlink()

    conn = init_db(str(db_path))
    source = LocalFolderSource(demo_dir)

    run_stage0(source, conn, cfg)
    run_stage1(source, conn, cfg)
    run_embeddings(source, conn, cfg)
    run_stage2_clustering(conn, cfg, index_path=faiss_path)
    run_stage3_verification(conn, cfg, source=source)
    run_quality_pipeline(conn, cfg, source=source)

    clusters = conn.execute("SELECT id, representative_id FROM clusters ORDER BY id").fetchall()
    results = {}

    for c in clusters:
        cid = c["id"]
        members = conn.execute(
            """SELECT id, file_path, file_size, width, height, format, md5, phash,
                      status, cluster_id, is_keeper, is_low_confidence,
                      ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, quality_status,
                      jpeg_quality_est
               FROM images WHERE cluster_id = ? ORDER BY id""",
            (cid,)
        ).fetchall()
        if not members:
            continue

        id_to_m = {m["id"]: m for m in members}
        # Find reference member
        ref_members = [m for m in members if m["quality_status"] == "reference"]
        ref_m = ref_members[0] if ref_members else None

        cluster_res = {
            "members": members,
            "ref_member": ref_m,
            "modes": {},
        }

        for mode in ["conservative", "balanced", "aggressive"]:
            res = choose_keeper(members, mode=mode, config=cfg)
            k_m = id_to_m[res["keeper_id"]]
            cluster_res["modes"][mode] = {
                "keeper": k_m,
                "delete_safe": [id_to_m[i] for i in res["delete_safe_ids"]],
                "review": [id_to_m[i] for i in res["review_ids"]],
                "total_savings_bytes": res["total_savings_bytes"],
            }
        results[cid] = cluster_res

    conn.close()
    return results


def run_comparison():
    demo_dir = Path.home() / "dedupe_test" / "demo"
    print(f"=== DEMO RUN 1: Normal Names in {demo_dir} ===")
    setup_demo_folder(demo_dir)

    # Record md5 to original file name
    file_map_run1 = {}
    for p in demo_dir.glob("*.jpg"):
        md5_val = file_md5(p)
        file_map_run1[p.name] = (md5_val, p.stat().st_size)

    run1_results = execute_pipeline(demo_dir)

    # Map Run 1 clusters by scene name (from files in cluster)
    run1_by_scene = {}
    for cid, c_data in run1_results.items():
        # Find scene name from first member
        first_path = Path(c_data["members"][0]["file_path"]).name
        scene = first_path.split("__")[0].replace(".jpg", "")
        run1_by_scene[scene] = c_data

    print(f"Run 1 completed: {len(run1_by_scene)} scene clusters evaluated.")

    # ----------------------------------------------------
    print(f"\n=== DEMO RUN 2: Renaming all files to img_<random>.jpg ===")
    random.seed(42)  # For reproducible random names
    name_mapping = {}  # new_name -> old_name
    old_files = list(demo_dir.glob("*.jpg"))
    # Generate unique random names
    used_names = set()
    for p in old_files:
        while True:
            rnd_hex = f"{random.randint(0, 0xFFFFFFFF):08x}"
            new_name = f"img_{rnd_hex}.jpg"
            if new_name not in used_names:
                used_names.add(new_name)
                break
        name_mapping[new_name] = p.name
        p.rename(demo_dir / new_name)

    # Delete any lingering DB/FAISS
    for extra in ["demo.db", "demo.faiss"]:
        extra_path = demo_dir / extra
        if extra_path.exists():
            extra_path.unlink()

    run2_results = execute_pipeline(demo_dir)

    # Map Run 2 clusters by scene name using reverse name mapping
    run2_by_scene = {}
    for cid, c_data in run2_results.items():
        first_rand_name = Path(c_data["members"][0]["file_path"]).name
        orig_name = name_mapping[first_rand_name]
        scene = orig_name.split("__")[0].replace(".jpg", "")
        run2_by_scene[scene] = c_data

    print(f"Run 2 completed: {len(run2_by_scene)} scene clusters evaluated.\n")

    # ----------------------------------------------------
    # Compare Run 1 vs Run 2
    print("=" * 95)
    print("  COMPARISON OF RUN 1 (NORMAL NAMES) vs RUN 2 (RANDOM STRINGS)")
    print("=" * 95)

    all_scenes = sorted(run1_by_scene.keys())
    ref_mismatches = 0
    keeper_mismatches = 0

    header = f"{'Scene':<14} | {'Metric / Mode':<15} | {'Run 1 (Normal Name)':<30} | {'Run 2 (Random Name)':<30} | {'Match?'}"
    print(header)
    print("-" * 95)

    for scene in all_scenes:
        c1 = run1_by_scene[scene]
        c2 = run2_by_scene[scene]

        # 1. Reference comparison
        ref1 = c1["ref_member"]
        ref2 = c2["ref_member"]
        ref1_name = Path(ref1["file_path"]).name if ref1 else "None"
        ref2_rand_name = Path(ref2["file_path"]).name if ref2 else "None"
        ref2_orig_name = name_mapping.get(ref2_rand_name, "Unknown")
        ref_match = (ref1["md5"] == ref2["md5"]) if (ref1 and ref2) else False
        if not ref_match:
            ref_mismatches += 1
        print(f"{scene:<14} | {'Reference':<15} | {ref1_name:<30} | {f'{ref2_rand_name} ({ref2_orig_name})':<30} | {'YES' if ref_match else 'MISMATCH'}")

        # 2. Mode comparisons
        for mode in ["conservative", "balanced", "aggressive"]:
            k1 = c1["modes"][mode]["keeper"]
            k2 = c2["modes"][mode]["keeper"]
            k1_name = Path(k1["file_path"]).name
            k2_rand_name = Path(k2["file_path"]).name
            k2_orig_name = name_mapping.get(k2_rand_name, "Unknown")

            k1_md5 = k1["md5"]
            k2_md5 = k2["md5"]
            keeper_match = (k1_md5 == k2_md5)

            # Check delete-safe sets
            ds1_md5s = {m["md5"] for m in c1["modes"][mode]["delete_safe"]}
            ds2_md5s = {m["md5"] for m in c2["modes"][mode]["delete_safe"]}
            ds_match = (ds1_md5s == ds2_md5s)

            # Check review sets
            rv1_md5s = {m["md5"] for m in c1["modes"][mode]["review"]}
            rv2_md5s = {m["md5"] for m in c2["modes"][mode]["review"]}
            rv_match = (rv1_md5s == rv2_md5s)

            status = "YES" if keeper_match else "MISMATCH"
            if not keeper_match:
                keeper_mismatches += 1
            print(f"{'':<14} | {mode.capitalize() + ' Keeper':<15} | {k1_name:<30} | {f'{k2_rand_name} ({k2_orig_name})':<30} | {status}")

        print("-" * 95)

    print(f"\nFinal Summary:")
    print(f"  Reference Mismatches : {ref_mismatches} / {len(all_scenes)}")
    print(f"  Keeper Mismatches    : {keeper_mismatches} / {len(all_scenes) * 3}")
    if ref_mismatches == 0 and keeper_mismatches == 0:
        print("  RESULT: References and keepers for each mode are 100% IDENTICAL in both runs!")
    print("=" * 95)


if __name__ == "__main__":
    run_comparison()
