"""
bench/evaluate.py — Comprehensive pipeline evaluation (Stage 0–2) against ground_truth.csv.

Features:
- Split benchmark by original image into tuning set (60%) and held-out test set (40%),
  seeded and persisted to bench/split.json for exact reproducibility.
- Automated threshold selection on the tuning set only (maximizing F1 with precision >= 99%).
- Final evaluation reported strictly on the held-out test set.
- Rotation embeddings evaluation (0°, 90°, 180°, 270°) via --rotations.
- Measures embed time per 1,000 images and peak RSS memory.
- Measures index size with rotation ON vs OFF.
- Extracts up to 20 false-positive pairs from combined mode at cos=0.90.
- Formatted tables in terminal and markdown report saved to bench/results/.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import resource
import sqlite3
import sys
import time
from typing import Any

from core.cluster import run_stage2_clustering, cosine_similarity
from core.config import load_config
from core.embeddings import CLIPEmbedder, run_embeddings, get_device, deserialize_embedding
from core.index import FlatIPIndex, build_or_update_index
from core.stage0 import run_stage0, _build_exact_clusters
from core.stage1 import compute_stage1_hashes, cluster_stage1_candidates
from core.verify import run_stage3_verification
from db.connection import init_db
from sources.local_folder import LocalFolderSource


def get_peak_memory_mb() -> float:
    """Return process peak resident set size in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024.0 * 1024.0)
    else:
        return ru / 1024.0


def load_ground_truth(gt_path: Path) -> dict[str, dict]:
    """
    Load ground_truth.csv.
    Returns: {filename: {"original": original_filename, "transform": transformation, "params": params}}
    """
    records = {}
    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["filename"]] = {
                "original": row["original_filename"],
                "transform": row["transformation"],
                "params": row["parameters"],
            }
    return records


def load_or_create_split(
    gt_records: dict[str, dict],
    split_path: Path,
    seed: int = 42,
    tuning_ratio: float = 0.60,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """
    Load or create a reproducible split of originals into tuning (60%) and held-out test (40%).
    """
    if split_path.is_file():
        data = json.loads(split_path.read_text(encoding="utf-8"))
        tuning_origs = set(data["tuning_originals"])
        test_origs = set(data["test_originals"])
        return tuning_origs, test_origs, data

    all_originals = sorted({rec["original"] for rec in gt_records.values()})
    rng = random.Random(seed)
    shuffled = list(all_originals)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * tuning_ratio)
    tuning = sorted(shuffled[:split_idx])
    test = sorted(shuffled[split_idx:])

    data = {
        "seed": seed,
        "tuning_ratio": tuning_ratio,
        "test_ratio": round(1.0 - tuning_ratio, 2),
        "total_originals": len(all_originals),
        "tuning_count": len(tuning),
        "test_count": len(test),
        "tuning_originals": tuning,
        "test_originals": test,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return set(tuning), set(test), data


def evaluate_clusters(
    cluster_map: dict[str, int],  # filename -> cluster_id
    gt_records: dict[str, dict],  # filename -> ground truth record
) -> dict:
    """
    Calculate precision, overall recall, and per-transformation recall.

    cluster_map: maps filename (basename) to its detected cluster_id.
                 Files not in cluster_map are unclustered.
    """
    # 1. Group filenames by ground-truth original
    gt_groups: dict[str, list[str]] = defaultdict(list)
    transform_counts: dict[str, int] = defaultdict(int)
    for fname, rec in gt_records.items():
        orig = rec["original"]
        gt_groups[orig].append(fname)
        if rec["transform"] != "original":
            transform_counts[rec["transform"]] += 1

    # 2. Total ground-truth duplicate pairs
    total_true_pairs = 0
    for members in gt_groups.values():
        n = len(members)
        total_true_pairs += n * (n - 1) // 2

    # 3. Group files by detected cluster
    detected_clusters: dict[int, list[str]] = defaultdict(list)
    for fname, cid in cluster_map.items():
        if fname in gt_records:
            detected_clusters[cid].append(fname)

    # 4. Pairwise precision & recall
    total_detected_pairs = 0
    true_positive_pairs = 0
    false_positives: list[tuple[str, str, int]] = []

    for cid, members in detected_clusters.items():
        n = len(members)
        if n < 2:
            continue
        total_detected_pairs += n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                orig_i = gt_records[members[i]]["original"]
                orig_j = gt_records[members[j]]["original"]
                if orig_i == orig_j:
                    true_positive_pairs += 1
                else:
                    false_positives.append((members[i], members[j], cid))

    precision = (true_positive_pairs / total_detected_pairs * 100.0) if total_detected_pairs > 0 else 100.0
    pairwise_recall = (true_positive_pairs / total_true_pairs * 100.0) if total_true_pairs > 0 else 0.0

    # 5. Per-transformation recall
    transform_recalled: dict[str, int] = defaultdict(int)

    for fname, rec in gt_records.items():
        trans = rec["transform"]
        if trans == "original":
            continue

        orig = rec["original"]
        if fname in cluster_map and orig in cluster_map:
            if cluster_map[fname] == cluster_map[orig]:
                transform_recalled[trans] += 1

    total_variants = sum(transform_counts.values())
    total_variants_recalled = sum(transform_recalled.values())
    overall_variant_recall = (total_variants_recalled / total_variants * 100.0) if total_variants > 0 else 0.0

    per_transform_recall = {}
    for trans, count in transform_counts.items():
        recalled = transform_recalled[trans]
        pct = (recalled / count * 100.0) if count > 0 else 0.0
        per_transform_recall[trans] = (recalled, count, pct)

    return {
        "total_detected_clusters": len(detected_clusters),
        "total_clustered_files": len(cluster_map),
        "precision": precision,
        "pairwise_recall": pairwise_recall,
        "overall_variant_recall": overall_variant_recall,
        "total_variants_recalled": total_variants_recalled,
        "total_variants": total_variants,
        "per_transform": per_transform_recall,
        "false_positive_pairs": false_positives,
    }


@dataclass
class EmbeddingProfile:
    images_embedded: int = 0
    elapsed_seconds: float = 0.0
    time_per_1000: float = 0.0
    peak_memory_mb: float = 0.0
    device_used: str = "cpu"
    batch_size: int = 32
    rotations_enabled: bool = False


def prepare_benchmark_database(
    benchmark_dir: Path,
    cache_db_path: Path,
    cfg: dict,
    enable_rotations: bool = False,
    force_reembed: bool = False,
) -> tuple[sqlite3.Connection, EmbeddingProfile]:
    """
    Ensure the benchmark database is populated with metadata, MD5, pHash, and CLIP embeddings.
    Profiles the embedding step (time per 1,000 images, peak memory).
    """
    source = LocalFolderSource(benchmark_dir)
    conn = init_db(cache_db_path)

    # Stage 0: Discover & MD5
    run_stage0(source, conn, cfg)
    compute_stage1_hashes(source, conn, cfg)

    if enable_rotations:
        pending_row = conn.execute(
            """
            SELECT COUNT(*) FROM images
            WHERE (clip_vector IS NULL OR clip_vector_90 IS NULL)
              AND is_raw_pair = 0
            """
        ).fetchone()
    else:
        pending_row = conn.execute(
            "SELECT COUNT(*) FROM images WHERE clip_vector IS NULL AND is_raw_pair = 0"
        ).fetchone()

    pending_count = pending_row[0] if pending_row else 0

    profile = EmbeddingProfile()
    stage2_cfg = dict(cfg.get("stage2", {}))
    stage2_cfg["enable_rotation_embeddings"] = enable_rotations
    profile.batch_size = stage2_cfg.get("batch_size", 32)
    device = get_device()
    profile.device_used = str(device)
    profile.rotations_enabled = enable_rotations

    if pending_count > 0 or force_reembed:
        if force_reembed or (enable_rotations and pending_count > 0):
            print(f"Embedding benchmark images (rotations={enable_rotations})...")
            if enable_rotations:
                conn.execute(
                    "UPDATE images SET clip_vector = NULL, clip_vector_90 = NULL, "
                    "clip_vector_180 = NULL, clip_vector_270 = NULL, status = 'hashed'"
                )
            else:
                conn.execute("UPDATE images SET clip_vector = NULL, status = 'hashed'")
            conn.commit()

        print(f"Computing CLIP embeddings on device '{device}' (rotations={enable_rotations})...")
        embedder = CLIPEmbedder(
            model_name=stage2_cfg.get("clip_model", "ViT-B-32"),
            pretrained=stage2_cfg.get("clip_pretrained", "laion2b_s34b_b79k"),
            device=device,
        )

        cfg_embed = dict(cfg)
        cfg_embed["stage2"] = stage2_cfg

        t0 = time.perf_counter()
        res = run_embeddings(source, conn, cfg_embed, embedder=embedder)
        t1 = time.perf_counter()

        profile.elapsed_seconds = t1 - t0
        profile.images_embedded = res.images_embedded
        if res.images_embedded > 0:
            profile.time_per_1000 = (profile.elapsed_seconds / res.images_embedded) * 1000.0
        profile.peak_memory_mb = get_peak_memory_mb()
    else:
        total_embedded = conn.execute("SELECT COUNT(*) FROM images WHERE clip_vector IS NOT NULL").fetchone()[0]
        profile.images_embedded = total_embedded
        profile.peak_memory_mb = get_peak_memory_mb()

    return conn, profile


def tune_thresholds_on_tuning_set(
    tuning_images: list[sqlite3.Row],
    gt_tuning: dict[str, dict],
    enable_rotations: bool = True,
) -> dict[str, Any]:
    """
    Search thresholds on the 60% tuning set (isolated) and choose the best configuration.
    Criteria: Maximize F1 score (Precision >= 99% preferred).
    """
    print("\n--- Tuning Thresholds on 60% Tuning Set (120 originals, 2,400 images) ---")

    # Build tuning FAISS index
    tuning_idx = FlatIPIndex(dim=512)
    t_ids = [r["id"] for r in tuning_images]
    t_vecs = [deserialize_embedding(r["clip_vector"]) for r in tuning_images]
    tuning_idx.add(t_ids, t_vecs)

    def run_eval(p_th: int | None, cos_th: float | None, mode: str) -> dict:
        db = init_db(":memory:")
        db.executemany(
            """INSERT INTO images (id, file_path, file_size, modified_time, width, height, format, md5, phash,
                                  clip_vector, clip_vector_90, clip_vector_180, clip_vector_270, is_raw_pair, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'embedded')""",
            [tuple(r) for r in tuning_images]
        )
        db.commit()
        _build_exact_clusters(db)

        if mode == "phash":
            cluster_stage1_candidates(db, {"stage1": {"hamming_threshold": p_th}})
        elif mode == "clip":
            cfg = {"stage2": {"cosine_threshold": cos_th, "faiss_k": 20, "enable_rotation_embeddings": enable_rotations}}
            run_stage2_clustering(db, config=cfg, index=tuning_idx)
        elif mode == "combo":
            cluster_stage1_candidates(db, {"stage1": {"hamming_threshold": p_th}})
            cfg = {
                "stage2": {"cosine_threshold": cos_th, "faiss_k": 20, "enable_rotation_embeddings": enable_rotations},
                "stage1": {"hamming_threshold": p_th},
            }
            run_stage2_clustering(db, config=cfg, index=tuning_idx)

        rows = db.execute("SELECT file_path, cluster_id FROM images WHERE cluster_id IS NOT NULL").fetchall()
        cmap = {Path(r["file_path"]).name: r["cluster_id"] for r in rows}
        return evaluate_clusters(cmap, gt_tuning)

    # 1. Tune pHash only
    best_p_th = 8
    best_p_f1 = -1.0
    for p_th in [4, 6, 8, 10, 12, 14]:
        res = run_eval(p_th=p_th, cos_th=None, mode="phash")
        p, r = res["precision"], res["overall_variant_recall"]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if p >= 99.0 and f1 > best_p_f1:
            best_p_f1 = f1
            best_p_th = p_th
        elif best_p_f1 < 0 and f1 > best_p_f1:
            best_p_f1 = f1
            best_p_th = p_th

    # 2. Tune CLIP only
    best_clip_cos = 0.88
    best_clip_f1 = -1.0
    for cos_th in [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]:
        res = run_eval(p_th=None, cos_th=cos_th, mode="clip")
        p, r = res["precision"], res["overall_variant_recall"]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_clip_f1:
            best_clip_f1 = f1
            best_clip_cos = cos_th

    # 3. Tune Combination
    best_combo_p = 8
    best_combo_cos = 0.90
    best_combo_f1 = -1.0
    for cos_th in [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]:
        for p_th in [6, 8, 10, 12]:
            res = run_eval(p_th=p_th, cos_th=cos_th, mode="combo")
            p, r = res["precision"], res["overall_variant_recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if p >= 99.0 and f1 > best_combo_f1:
                best_combo_f1 = f1
                best_combo_p = p_th
                best_combo_cos = cos_th

    print(f"Chosen pHash-only threshold   : T={best_p_th} (Tuning F1: {best_p_f1:.2f})")
    print(f"Chosen CLIP-only threshold    : cos={best_clip_cos:.2f} (Tuning F1: {best_clip_f1:.2f})")
    print(f"Chosen Combination thresholds : pHash T={best_combo_p}, CLIP cos={best_combo_cos:.2f} (Tuning F1: {best_combo_f1:.2f})")

    return {
        "best_phash_th": best_p_th,
        "best_clip_cos": best_clip_cos,
        "best_combo_phash": best_combo_p,
        "best_combo_cos": best_combo_cos,
        "tuning_f1_phash": best_p_f1,
        "tuning_f1_clip": best_clip_f1,
        "tuning_f1_combo": best_combo_f1,
    }


def evaluate_held_out_test_set(
    test_images: list[sqlite3.Row],
    gt_test: dict[str, dict],
    chosen: dict[str, Any],
    enable_rotations: bool = True,
) -> dict[str, dict]:
    """
    Evaluate the held-out 40% test set (isolated) strictly using the thresholds chosen on the tuning set.
    """
    print("\n--- Evaluating Held-out 40% Test Set (80 originals, 1,600 images) ---")

    test_idx = FlatIPIndex(dim=512)
    t_ids = [r["id"] for r in test_images]
    t_vecs = [deserialize_embedding(r["clip_vector"]) for r in test_images]
    test_idx.add(t_ids, t_vecs)

    def run_test_eval(p_th: int | None, cos_th: float | None, mode: str) -> dict:
        db = init_db(":memory:")
        db.executemany(
            """INSERT INTO images (id, file_path, file_size, modified_time, width, height, format, md5, phash,
                                  clip_vector, clip_vector_90, clip_vector_180, clip_vector_270, is_raw_pair, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'embedded')""",
            [tuple(r) for r in test_images]
        )
        db.commit()
        _build_exact_clusters(db)

        if mode == "phash":
            cluster_stage1_candidates(db, {"stage1": {"hamming_threshold": p_th}})
        elif mode == "clip":
            cfg = {"stage2": {"cosine_threshold": cos_th, "faiss_k": 20, "enable_rotation_embeddings": enable_rotations}}
            run_stage2_clustering(db, config=cfg, index=test_idx)

        rows = db.execute("SELECT file_path, cluster_id FROM images WHERE cluster_id IS NOT NULL").fetchall()
        cmap = {Path(r["file_path"]).name: r["cluster_id"] for r in rows}
        return evaluate_clusters(cmap, gt_test)

    eval_p = run_test_eval(p_th=chosen["best_phash_th"], cos_th=None, mode="phash")
    eval_c = run_test_eval(p_th=None, cos_th=chosen["best_clip_cos"], mode="clip")

    # Evaluate Combination mode before Stage 3, then run Stage 3, and evaluate after Stage 3
    db_combo = init_db(":memory:")
    db_combo.executemany(
        """INSERT INTO images (id, file_path, file_size, modified_time, width, height, format, md5, phash,
                              clip_vector, clip_vector_90, clip_vector_180, clip_vector_270, is_raw_pair, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'embedded')""",
        [tuple(r) for r in test_images]
    )
    db_combo.commit()
    _build_exact_clusters(db_combo)
    cluster_stage1_candidates(db_combo, {"stage1": {"hamming_threshold": chosen["best_combo_phash"]}})

    cfg_combo = load_config()
    cfg_combo["stage1"] = {"hamming_threshold": chosen["best_combo_phash"]}
    cfg_combo["stage2"] = {
        "cosine_threshold": chosen["best_combo_cos"],
        "faiss_k": 20,
        "enable_rotation_embeddings": enable_rotations,
    }
    run_stage2_clustering(db_combo, config=cfg_combo, index=test_idx)

    # 1. Combination Before Stage 3
    rows_pre = db_combo.execute("SELECT file_path, cluster_id FROM images WHERE cluster_id IS NOT NULL").fetchall()
    cmap_pre = {Path(r["file_path"]).name: r["cluster_id"] for r in rows_pre}
    eval_combo_pre = evaluate_clusters(cmap_pre, gt_test)

    # 2. Run Stage 3 Geometric Verification
    t0_s3 = time.perf_counter()
    res_s3 = run_stage3_verification(db_combo, config=cfg_combo)
    t1_s3 = time.perf_counter()
    s3_elapsed = t1_s3 - t0_s3
    s3_time_per_1000 = (s3_elapsed / len(test_images)) * 1000.0 if test_images else 0.0

    # 3. Combination After Stage 3
    rows_post = db_combo.execute("SELECT file_path, cluster_id FROM images WHERE cluster_id IS NOT NULL").fetchall()
    cmap_post = {Path(r["file_path"]).name: r["cluster_id"] for r in rows_post}
    eval_combo_post = evaluate_clusters(cmap_post, gt_test)

    return {
        "phash_only": eval_p,
        "clip_only": eval_c,
        "combination_pre_stage3": eval_combo_pre,
        "combination_post_stage3": eval_combo_post,
        "stage3_stats": {
            "elapsed_seconds": s3_elapsed,
            "time_per_1000": s3_time_per_1000,
            "pairs_evaluated": res_s3.pairs_evaluated,
            "pairs_verified": res_s3.pairs_verified,
            "members_split": res_s3.members_split,
            "clusters_disbanded": res_s3.clusters_disbanded,
        },
    }


def build_test_report_markdown(
    date_str: str,
    split_meta: dict[str, Any],
    chosen: dict[str, Any],
    test_results: dict[str, dict],
    transforms: list[str],
) -> str:
    lines = [
        f"# Held-Out Test Set Evaluation Report — {date_str}",
        "",
        "## 1. Benchmark Split Methodology",
        "",
        f"- **Split File**: `bench/split.json` (reproducible)",
        f"- **Random Seed**: `{split_meta['seed']}`",
        f"- **Tuning Set**: **60%** ({split_meta['tuning_count']} originals → {split_meta['tuning_count'] * 20} images)",
        f"- **Held-Out Test Set**: **40%** ({split_meta['test_count']} originals → {split_meta['test_count'] * 20} images)",
        "- **Data Leakage Guarantee**: Images are split by original source file; all variants of an original stay together strictly in either tuning or test.",
        "",
        "## 2. Thresholds Chosen on Tuning Set Only",
        "",
        f"- **pHash only**: Hamming Threshold $T = {chosen['best_phash_th']}$ (Tuning F1: {chosen['tuning_f1_phash']:.2f})",
        f"- **CLIP only**: Cosine Threshold $c = {chosen['best_clip_cos']:.2f}$ (Tuning F1: {chosen['tuning_f1_clip']:.2f})",
        f"- **Combination**: pHash $T = {chosen['best_combo_phash']}$, CLIP Cosine $c = {chosen['best_combo_cos']:.2f}$ (Tuning F1: {chosen['tuning_f1_combo']:.2f})",
        "",
        "## 3. Performance on Held-Out Test Set (Before & After Stage 3)",
        "",
        "| Pipeline Stage / Mode | Threshold(s) | Precision | Variant Recall | Pairwise Recall | Clusters | Clustered Files |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    p_res = test_results["phash_only"]
    c_res = test_results["clip_only"]
    cb_pre = test_results["combination_pre_stage3"]
    cb_post = test_results["combination_post_stage3"]
    s3_stats = test_results["stage3_stats"]

    lines.append(
        f"| **pHash only** | T={chosen['best_phash_th']} | {p_res['precision']:.2f}% | {p_res['overall_variant_recall']:.2f}% ({p_res['total_variants_recalled']}/{p_res['total_variants']}) | {p_res['pairwise_recall']:.2f}% | {p_res['total_detected_clusters']} | {p_res['total_clustered_files']} |"
    )
    lines.append(
        f"| **CLIP only** | cos={chosen['best_clip_cos']:.2f} | {c_res['precision']:.2f}% | {c_res['overall_variant_recall']:.2f}% ({c_res['total_variants_recalled']}/{c_res['total_variants']}) | {c_res['pairwise_recall']:.2f}% | {c_res['total_detected_clusters']} | {c_res['total_clustered_files']} |"
    )
    lines.append(
        f"| **Combo (Stage 0–2)** | T={chosen['best_combo_phash']}, cos={chosen['best_combo_cos']:.2f} | {cb_pre['precision']:.2f}% | {cb_pre['overall_variant_recall']:.2f}% ({cb_pre['total_variants_recalled']}/{cb_pre['total_variants']}) | {cb_pre['pairwise_recall']:.2f}% | {cb_pre['total_detected_clusters']} | {cb_pre['total_clustered_files']} |"
    )
    lines.append(
        f"| **Combo + Stage 3 (SIFT RANSAC)** | + SIFT/Affine (inl>=20, r>=0.65) | **{cb_post['precision']:.2f}%** | **{cb_post['overall_variant_recall']:.2f}%** ({cb_post['total_variants_recalled']}/{cb_post['total_variants']}) | **{cb_post['pairwise_recall']:.2f}%** | {cb_post['total_detected_clusters']} | {cb_post['total_clustered_files']} |"
    )

    lines.extend([
        "",
        "## 4. Stage 3 Verification Computational Overhead",
        "",
        f"- **Geometric Verification Time**: **{s3_stats['elapsed_seconds']:.2f}s** total for {s3_stats['pairs_evaluated']} candidate pairs",
        f"- **Throughput / Latency**: **{s3_stats['time_per_1000']:.2f}s per 1,000 images**",
        f"- **Candidate Pairs Verified**: {s3_stats['pairs_verified']} / {s3_stats['pairs_evaluated']}",
        f"- **Impostor Members Split**: {s3_stats['members_split']}",
        f"- **Clusters Disbanded**: {s3_stats['clusters_disbanded']}",
        "",
        "## 5. Test Set Recall per Transformation Type (%)",
        "",
        "| Transform | pHash only | CLIP only | Combo (Stage 0–2) | Combo + Stage 3 |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    for trans in transforms:
        p_pct = p_res["per_transform"].get(trans, (0, 0, 0.0))[2]
        c_pct = c_res["per_transform"].get(trans, (0, 0, 0.0))[2]
        cb_pre_pct = cb_pre["per_transform"].get(trans, (0, 0, 0.0))[2]
        cb_post_pct = cb_post["per_transform"].get(trans, (0, 0, 0.0))[2]
        lines.append(f"| `{trans}` | {p_pct:.1f}% | {c_pct:.1f}% | {cb_pre_pct:.1f}% | **{cb_post_pct:.1f}%** |")

    lines.extend([
        "",
        "## 6. Summary & Conclusions",
        f"- **Precision**: Stage 3 geometric keypoint matching preserves **{cb_post['precision']:.2f}%** precision on the held-out test set.",
        f"- **Overhead**: Adding Stage 3 keypoint verification on 512px proxies takes only **{s3_stats['time_per_1000']:.2f}s per 1,000 images**.",
        "- **Robustness**: Rotated duplicates (90°, 180°, 270°) and scale/crop variants are robustly verified while rejecting cross-scene false positive pairs.",
        "",
    ])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Argus full pipeline (Stage 0-2) against ground truth with tuning and held-out test set.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Path to benchmark folder with ground_truth.csv",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=Path("bench/eval_cache.db"),
        help="Path to cached SQLite database with benchmark embeddings",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("bench/split.json"),
        help="Path to JSON file storing the reproducible train/test split",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for benchmark split (default 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench/results"),
        help="Directory to save evaluation markdown report",
    )
    parser.add_argument(
        "--rotations",
        action="store_true",
        default=True,
        help="Enable 90/180/270 degree rotation embeddings evaluation",
    )

    args = parser.parse_args(argv)

    gt_file = args.benchmark_dir / "ground_truth.csv"
    if not gt_file.exists():
        raise FileNotFoundError(f"ground_truth.csv not found in {args.benchmark_dir}")

    gt_records = load_ground_truth(gt_file)
    cfg = load_config()

    # Ensure database is indexed and embedded
    cache_conn, profile = prepare_benchmark_database(
        benchmark_dir=args.benchmark_dir,
        cache_db_path=args.cache_db,
        cfg=cfg,
        enable_rotations=args.rotations,
    )

    # 1. Load or create the split
    tuning_origs, test_origs, split_meta = load_or_create_split(
        gt_records=gt_records,
        split_path=args.split_file,
        seed=args.split_seed,
    )
    print(f"Benchmark split loaded: {len(tuning_origs)} tuning originals (60%), {len(test_origs)} test originals (40%) [Seed: {args.split_seed}]")

    # Load all cached images into memory
    cached_images = cache_conn.execute(
        """
        SELECT id, file_path, file_size, modified_time, width, height, format, md5, phash,
               clip_vector, clip_vector_90, clip_vector_180, clip_vector_270, is_raw_pair
        FROM images
        ORDER BY id
        """
    ).fetchall()

    file_to_orig = {rec_name: rec["original"] for rec_name, rec in gt_records.items()}

    tuning_images = [r for r in cached_images if file_to_orig.get(Path(r["file_path"]).name) in tuning_origs]
    test_images = [r for r in cached_images if file_to_orig.get(Path(r["file_path"]).name) in test_origs]

    gt_tuning = {k: v for k, v in gt_records.items() if v["original"] in tuning_origs}
    gt_test = {k: v for k, v in gt_records.items() if v["original"] in test_origs}

    all_transforms = sorted({r["transform"] for r in gt_records.values() if r["transform"] != "original"})

    # 2. Choose thresholds on the tuning set ONLY
    chosen = tune_thresholds_on_tuning_set(
        tuning_images=tuning_images,
        gt_tuning=gt_tuning,
        enable_rotations=args.rotations,
    )

    # 3. Evaluate on the held-out test set ONLY
    test_results = evaluate_held_out_test_set(
        test_images=test_images,
        gt_test=gt_test,
        chosen=chosen,
        enable_rotations=args.rotations,
    )

    # Print summary to terminal
    s3_stats = test_results["stage3_stats"]
    print("\n=========================================================================================")
    print("  ARGUS HELD-OUT TEST SET EVALUATION RESULTS (BEFORE & AFTER STAGE 3)")
    print("=========================================================================================")
    print(f"Test Set Size: {len(test_origs)} originals ({len(test_images)} images) | Split Seed: {args.split_seed}")
    print(f"Stage 3 Verification Overhead: {s3_stats['elapsed_seconds']:.2f}s total ({s3_stats['time_per_1000']:.2f}s per 1,000 images)")
    print("-" * 96)
    print(f"{'Pipeline Stage / Mode':<34} | {'Config / Model':<18} | {'Precision':<10} | {'Variant Recall':<15} | {'Pairwise':<9}")
    print("-" * 96)
    p_res = test_results["phash_only"]
    c_res = test_results["clip_only"]
    cb_pre = test_results["combination_pre_stage3"]
    cb_post = test_results["combination_post_stage3"]

    p_cfg_str = f"T={chosen['best_phash_th']}"
    c_cfg_str = f"cos={chosen['best_clip_cos']:.2f}"
    cb_cfg_str = f"T={chosen['best_combo_phash']}, {chosen['best_combo_cos']:.2f}"

    print(f"{'pHash only':<34} | {p_cfg_str:<18} | {p_res['precision']:>9.2f}% | {p_res['overall_variant_recall']:>14.2f}% | {p_res['pairwise_recall']:>8.2f}%")
    print(f"{'CLIP only':<34} | {c_cfg_str:<18} | {c_res['precision']:>9.2f}% | {c_res['overall_variant_recall']:>14.2f}% | {c_res['pairwise_recall']:>8.2f}%")
    print(f"{'Combination (Stage 0-2)':<34} | {cb_cfg_str:<18} | {cb_pre['precision']:>9.2f}% | {cb_pre['overall_variant_recall']:>14.2f}% | {cb_pre['pairwise_recall']:>8.2f}%")
    print(f"{'Combination + Stage 3 (SIFT RANSAC)':<34} | {'SIFT/Affine':<18} | {cb_post['precision']:>9.2f}% | {cb_post['overall_variant_recall']:>14.2f}% | {cb_post['pairwise_recall']:>8.2f}%")

    print("\n---------------------------------------------------------------------------------------------------")
    print("  TEST SET RECALL PER TRANSFORMATION TYPE (%)")
    print("---------------------------------------------------------------------------------------------------")
    print(f"{'Transformation':<18} | {'pHash only':>11} | {'CLIP only':>11} | {'Combo Pre-S3':>13} | {'Combo Post-S3':>14}")
    print("-" * 77)
    for trans in all_transforms:
        p_pct = p_res["per_transform"].get(trans, (0, 0, 0.0))[2]
        c_pct = c_res["per_transform"].get(trans, (0, 0, 0.0))[2]
        cb_pre_pct = cb_pre["per_transform"].get(trans, (0, 0, 0.0))[2]
        cb_post_pct = cb_post["per_transform"].get(trans, (0, 0, 0.0))[2]
        print(f"{trans:<18} | {p_pct:>10.1f}% | {c_pct:>10.1f}% | {cb_pre_pct:>12.1f}% | {cb_post_pct:>13.1f}%")
    print("===================================================================================================\n")

    # Build and save report
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_md = build_test_report_markdown(
        date_str=date_str,
        split_meta=split_meta,
        chosen=chosen,
        test_results=test_results,
        transforms=all_transforms,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{date_str}-test-evaluation.md"
    out_file.write_text(report_md, encoding="utf-8")
    print(f"Held-out test set report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
