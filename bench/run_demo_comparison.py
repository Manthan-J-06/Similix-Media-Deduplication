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
from core.quality import run_quality_pipeline, select_reference_member
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

    clusters = conn.execute("SELECT id, representative_id, reference_uncertain FROM clusters ORDER BY id").fetchall()
    results = {}

    for c in clusters:
        cid = c["id"]
        members = conn.execute(
            """SELECT id, file_path, file_size, width, height, format, md5, phash,
                      status, cluster_id, is_keeper, is_low_confidence,
                      ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, quality_status,
                      source_likelihood, jpeg_quality_est
               FROM images WHERE cluster_id = ? ORDER BY id""",
            (cid,)
        ).fetchall()
        if not members:
            continue

        id_to_m = {m["id"]: m for m in members}
        # Find reference member
        ref_members = [m for m in members if m["quality_status"] == "reference"]
        ref_m = ref_members[0] if ref_members else None

        ref_unc = c["reference_uncertain"] if "reference_uncertain" in c.keys() else None
        ref_details = select_reference_member([dict(m) for m in members], source=source)
        gap = ref_details.get("top_candidate_gap")
        h2h = ref_details.get("head_to_head_result")

        cluster_res = {
            "members": members,
            "ref_member": ref_m,
            "reference_uncertain": ref_unc,
            "top_candidate_gap": gap,
            "head_to_head_result": h2h,
            "modes": {},
        }

        for mode in ["conservative", "balanced", "aggressive"]:
            res = choose_keeper(members, mode=mode, config=cfg, reference_uncertain=ref_unc)
            k_m = id_to_m[res["keeper_id"]]
            cluster_res["modes"][mode] = {
                "keeper": k_m,
                "delete_safe": [id_to_m[i] for i in res["delete_safe_ids"]],
                "review": [id_to_m[i] for i in res["review_ids"]],
                "total_savings_bytes": res["total_savings_bytes"],
                "reasons": res["reasons"],
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
    true_original_references = 0

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

        is_true_original = (ref1_name == f"{scene}.jpg" or ref2_orig_name == f"{scene}.jpg" or "exact_copy" in ref1_name)
        if is_true_original:
            true_original_references += 1

        print(f"{scene:<14} | {'Reference':<15} | {ref1_name:<30} | {f'{ref2_rand_name} ({ref2_orig_name})':<30} | {'YES' if ref_match else 'MISMATCH'}")
        unc1 = c1.get("reference_uncertain", 0)
        unc2 = c2.get("reference_uncertain", 0)
        unc_match = (unc1 == unc2)
        print(f"{'':<14} | {'Ref Uncertain':<15} | {str(unc1):<30} | {str(unc2):<30} | {'YES' if unc_match else 'MISMATCH'}")

        gap1 = c1.get("top_candidate_gap")
        gap2 = c2.get("top_candidate_gap")
        gap1_str = f"{gap1:.6f}" if gap1 is not None else "N/A"
        gap2_str = f"{gap2:.6f}" if gap2 is not None else "N/A"
        gap_match = (gap1 == gap2)
        print(f"{'':<14} | {'Top Cand Gap':<15} | {gap1_str:<30} | {gap2_str:<30} | {'YES' if gap_match else 'MISMATCH'}")

        h2h1 = c1.get("head_to_head_result") or "N/A"
        h2h2 = c2.get("head_to_head_result") or "N/A"
        print(f"{'':<14} | {'Head-to-Head':<15} | {h2h1:<30} | {h2h2:<30} |")

        # Print Source-Likelihood Scores
        print(f"{'':<14} | {'Source Scores':<15} | {'(Run 1 member scores below)':<30} | {'(Run 2 member scores below)':<30} |")
        sorted_m1 = sorted([dict(m) for m in c1["members"]], key=lambda m: (m.get("source_likelihood") if m.get("source_likelihood") is not None else -1.0), reverse=True)
        id_to_m2 = {m["md5"]: dict(m) for m in c2["members"]}
        for m in sorted_m1:
            m_name = Path(m["file_path"]).name
            s1 = m.get("source_likelihood")
            s1_str = f"{s1:.6f}" if s1 is not None else "N/A"
            m2 = id_to_m2.get(m["md5"])
            if m2:
                m2_rand = Path(m2["file_path"]).name
                s2 = m2.get("source_likelihood")
                s2_str = f"{s2:.6f}" if s2 is not None else "N/A"
                m2_disp = f"{m2_rand}: {s2_str}"
            else:
                m2_disp = "N/A"
            print(f"{'':<14} | {'':<15} | {f'{m_name}: {s1_str}':<30} | {m2_disp:<30} |")

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

            status = "YES" if keeper_match else "MISMATCH"
            if not keeper_match:
                keeper_mismatches += 1
            print(f"{'':<14} | {mode.capitalize() + ' Keeper':<15} | {k1_name:<30} | {f'{k2_rand_name} ({k2_orig_name})':<30} | {status}")

            # Delete-safe list
            ds1_names = [Path(m["file_path"]).name for m in c1["modes"][mode]["delete_safe"]]
            ds2_rand_names = [Path(m["file_path"]).name for m in c2["modes"][mode]["delete_safe"]]
            ds2_orig_names = [name_mapping.get(n, "Unknown") for n in ds2_rand_names]
            ds_match = (sorted([m["md5"] for m in c1["modes"][mode]["delete_safe"]]) == sorted([m["md5"] for m in c2["modes"][mode]["delete_safe"]]))
            ds1_str = ", ".join(ds1_names) if ds1_names else "(none)"
            ds2_str = ", ".join(ds2_orig_names) if ds2_orig_names else "(none)"
            print(f"{'':<14} | {mode.capitalize()[:4] + ' DelSafe':<15} | {ds1_str:<30} | {ds2_str:<30} | {'YES' if ds_match else 'MISMATCH'}")

        print("-" * 95)

    print(f"\nFinal Summary:")
    print(f"  Reference Mismatches : {ref_mismatches} / {len(all_scenes)}")
    print(f"  Keeper Mismatches    : {keeper_mismatches} / {len(all_scenes) * 3}")
    print(f"  True Original References: {true_original_references} / {len(all_scenes)}")
    if ref_mismatches == 0 and keeper_mismatches == 0 and true_original_references == len(all_scenes):
        print("  RESULT: The true original is the reference in all 6 scenes!")
        print("  References and keepers for each mode are 100% IDENTICAL across normal and random names!")
    print("=" * 95)


if __name__ == "__main__":
    run_comparison()
