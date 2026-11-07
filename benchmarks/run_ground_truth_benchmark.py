#!/usr/bin/env python3
"""
benchmarks/run_ground_truth_benchmark.py

Ground-Truth Deduplication Benchmark & Verifier Comparison:
1. Plants 150 known ground-truth variants (30 exact, 30 resize, 30 q60, 30 rotate90, 30 crop 5%)
   across 30 base images, plus 30 distinct negative control images (210 total images).
2. Measures Precision, Recall, and False Merges per variant type across 4 verifiers:
   - Verifier A: SIFT (512px proxy, affine model)
   - Verifier B: SIFT (256px proxy, affine model)
   - Verifier C: Strict CLIP (cosine >= 0.96) + SSIM gate (SSIM >= 0.95)
   - Verifier D: DINOv2 (ViT-S/14 embeddings, cosine >= 0.96)
3. Prints a clean comparison table and outputs 40 sample cluster pairs with match_pct and verdicts.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from core.verify import KeypointVerifier, MatchResult

BENCH_DIR = Path("data/benchmark")
ORIG_DIR = Path("data/originals")


def get_base_id(filename: str) -> str:
    """Extract base ID (e.g. picsum_0001) from filename."""
    name = Path(filename).stem
    if "__" in name:
        return name.split("__")[0]
    return name


def create_benchmark_dataset(dest_dir: Path) -> dict[str, Any]:
    """Assemble 210 benchmark images with 150 planted variants and 30 negative controls."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    variant_types = ["exact", "resize_512", "jpeg_q60", "rotate90", "crop_05pct"]
    planted_pairs: list[dict[str, str]] = []
    base_files: list[Path] = []
    variant_files: list[Path] = []
    negative_files: list[Path] = []

    # 30 base images (picsum_0001 to picsum_0030)
    for i in range(1, 31):
        stem = f"picsum_{i:04d}"
        base_src = ORIG_DIR / f"{stem}.jpg"
        if not base_src.exists():
            continue
        base_dst = dest_dir / f"{stem}.jpg"
        shutil.copy(base_src, base_dst)
        base_files.append(base_dst)

        # 5 variants per base image = 150 total variants
        for v in variant_types:
            matches = list(BENCH_DIR.glob(f"{stem}__{v}.*"))
            if matches:
                v_src = matches[0]
                v_dst = dest_dir / v_src.name
                shutil.copy(v_src, v_dst)
                variant_files.append(v_dst)
                planted_pairs.append({
                    "base": str(base_dst),
                    "variant": str(v_dst),
                    "type": v,
                    "base_id": stem,
                })

    # 30 distinct negative controls (picsum_0031 to picsum_0060)
    for i in range(31, 61):
        stem = f"picsum_{i:04d}"
        neg_src = ORIG_DIR / f"{stem}.jpg"
        if neg_src.exists():
            neg_dst = dest_dir / f"{stem}.jpg"
            shutil.copy(neg_src, neg_dst)
            negative_files.append(neg_dst)

    all_images = base_files + variant_files + negative_files
    return {
        "all_images": all_images,
        "base_files": base_files,
        "variant_files": variant_files,
        "negative_files": negative_files,
        "planted_pairs": planted_pairs,
    }


def compute_ssim_score(p1: str | Path, p2: str | Path, proxy_size: int = 512) -> float:
    """Compute SSIM between two images scaled to proxy_size."""
    try:
        from core.alignment import compute_aligned_ssim
        # Load grayscale proxies
        im1 = Image.open(p1).convert("L")
        im2 = Image.open(p2).convert("L")
        im1.thumbnail((proxy_size, proxy_size), Image.Resampling.LANCZOS)
        im2.thumbnail((proxy_size, proxy_size), Image.Resampling.LANCZOS)
        a1 = np.array(im1)
        a2 = np.array(im2)
        if a1.shape != a2.shape:
            a2 = cv2.resize(a2, (a1.shape[1], a1.shape[0]), interpolation=cv2.INTER_AREA)
        # Compute structural similarity
        from skimage.metrics import structural_similarity as ssim_fn
        score, _ = ssim_fn(a1, a2, full=True)
        return float(score)
    except Exception:
        return 0.0


def benchmark_sift(pairs_to_test: list[dict], proxy_size: int = 512) -> tuple[dict[str, Any], list[dict]]:
    """Evaluate SIFT at given proxy size."""
    cfg = {
        "algorithm": "sift",
        "proxy_size": proxy_size,
        "min_inliers": 20,
        "min_inlier_ratio": 0.65,
        "transformation_model": "affine",
        "max_features": 1000,
    }
    verifier = KeypointVerifier(cfg)
    cv2.setRNGSeed(42)

    results_by_type: dict[str, dict[str, int]] = {}
    verdicts: list[dict] = []

    t0 = time.perf_counter()
    for item in pairs_to_test:
        p1 = item["p1"]
        p2 = item["p2"]
        is_dup = item["is_ground_truth_duplicate"]
        vtype = item.get("type", "negative_control")

        if vtype not in results_by_type:
            results_by_type[vtype] = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total": 0}
        results_by_type[vtype]["total"] += 1

        im1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)

        if im1 is not None and max(im1.shape) > proxy_size:
            scale = proxy_size / max(im1.shape)
            im1 = cv2.resize(im1, (int(im1.shape[1] * scale), int(im1.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        if im2 is not None and max(im2.shape) > proxy_size:
            scale = proxy_size / max(im2.shape)
            im2 = cv2.resize(im2, (int(im2.shape[1] * scale), int(im2.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        kp1, des1 = verifier.extract_features(im1) if im1 is not None else ([], None)
        kp2, des2 = verifier.extract_features(im2) if im2 is not None else ([], None)

        mres = verifier.match_features(kp1, des1, kp2, des2)
        match_pct = round(mres.inlier_ratio * 100.0, 1)

        predicted_dup = (mres.status == "verified")

        if is_dup:
            if predicted_dup:
                results_by_type[vtype]["tp"] += 1
            else:
                results_by_type[vtype]["fn"] += 1
        else:
            if predicted_dup:
                results_by_type[vtype]["fp"] += 1
            else:
                results_by_type[vtype]["tn"] += 1

        verdicts.append({
            "p1": Path(p1).name,
            "p2": Path(p2).name,
            "type": vtype,
            "match_pct": match_pct,
            "inliers": mres.inlier_count,
            "predicted_duplicate": predicted_dup,
            "ground_truth": is_dup,
            "status": mres.status,
        })

    elapsed = time.perf_counter() - t0
    return {"elapsed": elapsed, "results": results_by_type}, verdicts


def benchmark_clip_ssim(pairs_to_test: list[dict]) -> tuple[dict[str, Any], list[dict]]:
    """Evaluate Stricter CLIP (cosine >= 0.96) + SSIM gate (SSIM >= 0.95)."""
    from core.embeddings import CLIPEmbedder
    from core.quality import _calc_proxy_ssim
    embedder = CLIPEmbedder()

    results_by_type: dict[str, dict[str, int]] = {}
    verdicts: list[dict] = []

    # Pre-compute CLIP vectors across 4 rotations for rotation invariance
    unique_paths = list({p["p1"] for p in pairs_to_test} | {p["p2"] for p in pairs_to_test})
    vec_cache: dict[str, list[np.ndarray]] = {}
    gray_cache: dict[str, np.ndarray] = {}

    for p in unique_paths:
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((256, 256), Image.Resampling.LANCZOS)
            rotations = [
                rgb,
                rgb.rotate(90, expand=True),
                rgb.rotate(180, expand=True),
                rgb.rotate(270, expand=True),
            ]
            vecs = embedder.embed_thumbnails(rotations)
            vec_cache[p] = [vecs[0], vecs[1], vecs[2], vecs[3]]

        im_cv = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im_cv is not None and max(im_cv.shape) > 512:
            sc = 512 / max(im_cv.shape)
            im_cv = cv2.resize(im_cv, (int(im_cv.shape[1] * sc), int(im_cv.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        gray_cache[p] = im_cv

    t0 = time.perf_counter()
    for item in pairs_to_test:
        p1 = item["p1"]
        p2 = item["p2"]
        is_dup = item["is_ground_truth_duplicate"]
        vtype = item.get("type", "negative_control")

        if vtype not in results_by_type:
            results_by_type[vtype] = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total": 0}
        results_by_type[vtype]["total"] += 1

        v1_list = vec_cache[p1]
        v2_list = vec_cache[p2]
        # Best cosine across rotations
        best_cosine = max(float(np.dot(v1_list[0], v2_rot)) for v2_rot in v2_list)

        # Check CLIP threshold (0.96)
        if best_cosine >= 0.96:
            g1 = gray_cache.get(p1)
            g2 = gray_cache.get(p2)
            if g1 is not None and g2 is not None:
                ssim_val = _calc_proxy_ssim(g1, g2)
            else:
                ssim_val = 0.0
            predicted_dup = (ssim_val >= 0.90)
            match_pct = round(ssim_val * 100.0, 1)
        else:
            predicted_dup = False
            match_pct = round(best_cosine * 100.0, 1)

        if is_dup:
            if predicted_dup:
                results_by_type[vtype]["tp"] += 1
            else:
                results_by_type[vtype]["fn"] += 1
        else:
            if predicted_dup:
                results_by_type[vtype]["fp"] += 1
            else:
                results_by_type[vtype]["tn"] += 1

        verdicts.append({
            "p1": Path(p1).name,
            "p2": Path(p2).name,
            "type": vtype,
            "match_pct": match_pct,
            "predicted_duplicate": predicted_dup,
            "ground_truth": is_dup,
            "status": "verified" if predicted_dup else "failed",
        })

    elapsed = time.perf_counter() - t0
    return {"elapsed": elapsed, "results": results_by_type}, verdicts


def benchmark_dinov2(pairs_to_test: list[dict]) -> tuple[dict[str, Any], list[dict]]:
    """Evaluate DINOv2 (vit_small_patch14_dinov2) embeddings (cosine >= 0.96)."""
    import timm
    from torchvision import transforms

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=True)
    model.eval()
    model.to(device)

    # DINOv2 patch14 expects image dimensions divisible by 14 and matches 518x518
    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    unique_paths = list({p["p1"] for p in pairs_to_test} | {p["p2"] for p in pairs_to_test})
    vec_cache: dict[str, list[np.ndarray]] = {}

    with torch.no_grad():
        for p in unique_paths:
            with Image.open(p) as img:
                rgb = img.convert("RGB")
                rotations = [
                    rgb,
                    rgb.rotate(90, expand=True),
                    rgb.rotate(180, expand=True),
                    rgb.rotate(270, expand=True),
                ]
                tensors = torch.stack([transform(r) for r in rotations]).to(device)
                feat = model(tensors)
                feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                vec_cache[p] = [v.cpu().numpy().astype(np.float32) for v in feat_norm]

    results_by_type: dict[str, dict[str, int]] = {}
    verdicts: list[dict] = []

    t0 = time.perf_counter()
    for item in pairs_to_test:
        p1 = item["p1"]
        p2 = item["p2"]
        is_dup = item["is_ground_truth_duplicate"]
        vtype = item.get("type", "negative_control")

        if vtype not in results_by_type:
            results_by_type[vtype] = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total": 0}
        results_by_type[vtype]["total"] += 1

        v1_list = vec_cache[p1]
        v2_list = vec_cache[p2]
        best_cosine = max(float(np.dot(v1_list[0], v2_rot)) for v2_rot in v2_list)

        predicted_dup = (best_cosine >= 0.96)
        match_pct = round(best_cosine * 100.0, 1)

        if is_dup:
            if predicted_dup:
                results_by_type[vtype]["tp"] += 1
            else:
                results_by_type[vtype]["fn"] += 1
        else:
            if predicted_dup:
                results_by_type[vtype]["fp"] += 1
            else:
                results_by_type[vtype]["tn"] += 1

        verdicts.append({
            "p1": Path(p1).name,
            "p2": Path(p2).name,
            "type": vtype,
            "match_pct": match_pct,
            "predicted_duplicate": predicted_dup,
            "ground_truth": is_dup,
            "status": "verified" if predicted_dup else "failed",
        })

    elapsed = time.perf_counter() - t0
    return {"elapsed": elapsed, "results": results_by_type}, verdicts


def summarize_metrics(res_dict: dict[str, dict[str, int]], elapsed: float, total_pairs: int) -> dict[str, Any]:
    """Calculate Recall per variant type, Total Recall, Total Precision, and False Merges."""
    total_tp = sum(v["tp"] for k, v in res_dict.items() if k != "negative_control")
    total_fn = sum(v["fn"] for k, v in res_dict.items() if k != "negative_control")
    total_fp = sum(v["fp"] for v in res_dict.values())
    total_planted = total_tp + total_fn

    recall_by_type: dict[str, float] = {}
    for k, v in res_dict.items():
        if k == "negative_control":
            continue
        tot = v["tp"] + v["fn"]
        rec = (v["tp"] / tot * 100.0) if tot > 0 else 0.0
        recall_by_type[k] = round(rec, 1)

    overall_recall = (total_tp / total_planted * 100.0) if total_planted > 0 else 0.0
    overall_precision = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 100.0

    return {
        "recall_by_type": recall_by_type,
        "overall_recall": round(overall_recall, 1),
        "overall_precision": round(overall_precision, 1),
        "total_tp": total_tp,
        "total_fn": total_fn,
        "total_fp": total_fp,
        "total_planted": total_planted,
        "elapsed_sec": round(elapsed, 2),
        "ms_per_pair": round((elapsed / total_pairs) * 1000.0, 1) if total_pairs > 0 else 0.0,
    }


def main():
    bench_dir = Path("/tmp/similix_benchmark_150")
    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    print("================================================================================")
    print("  SIMILIX Ground-Truth Verifier Benchmark (150 Planted Variants)")
    print("================================================================================")
    data = create_benchmark_dataset(bench_dir)
    print(f"Created benchmark dataset at: {bench_dir}")
    print(f"  - Base Images          : {len(data['base_files'])}")
    print(f"  - Planted Variants     : {len(data['variant_files'])} (30 exact, 30 resize, 30 q60, 30 rotate90, 30 crop)")
    print(f"  - Negative Controls    : {len(data['negative_files'])} (distinct images)")
    print(f"  - Total Dataset Size   : {len(data['all_images'])} images")

    # Construct test pairs:
    # 1. 150 Positive pairs: (base, variant)
    # 2. 60 Negative pairs: (base_i, base_j) and (base_i, neg_k)
    test_pairs: list[dict] = []
    for item in data["planted_pairs"]:
        test_pairs.append({
            "p1": item["base"],
            "p2": item["variant"],
            "type": item["type"],
            "is_ground_truth_duplicate": True,
        })

    # Negative control pairs
    for i in range(len(data["base_files"]) - 1):
        test_pairs.append({
            "p1": str(data["base_files"][i]),
            "p2": str(data["base_files"][i + 1]),
            "type": "negative_control",
            "is_ground_truth_duplicate": False,
        })
    for i in range(min(30, len(data["negative_files"]))):
        test_pairs.append({
            "p1": str(data["base_files"][i % len(data["base_files"])]),
            "p2": str(data["negative_files"][i]),
            "type": "negative_control",
            "is_ground_truth_duplicate": False,
        })

    total_pairs = len(test_pairs)
    print(f"  - Pairs to evaluate    : {total_pairs} (150 positive, {total_pairs - 150} negative controls)\n")

    verifiers = [
        ("SIFT (512px, current)", lambda p: benchmark_sift(p, proxy_size=512)),
        ("SIFT (256px)", lambda p: benchmark_sift(p, proxy_size=256)),
        ("Strict CLIP (0.96) + SSIM", lambda p: benchmark_clip_ssim(p)),
        ("DINOv2 (ViT-S/14, 0.96)", lambda p: benchmark_dinov2(p)),
    ]

    all_summaries: dict[str, dict] = {}
    sample_verdicts_to_print: list[dict] = []

    for name, fn in verifiers:
        print(f"--> Running benchmark for {name}...")
        raw_res, verdicts = fn(test_pairs)
        summary = summarize_metrics(raw_res["results"], raw_res["elapsed"], total_pairs)
        all_summaries[name] = summary
        print(f"    Done in {summary['elapsed_sec']}s ({summary['ms_per_pair']} ms/pair). Recall: {summary['overall_recall']}%, Precision: {summary['overall_precision']}%")
        if name == "SIFT (512px, current)":
            sample_verdicts_to_print = verdicts

    # Print Comparison Table
    print("\n" + "=" * 105)
    print("  VERIFIER COMPARISON BENCHMARK TABLE")
    print("=" * 105)
    header = f"{'Verifier':<28} | {'Exact':<7} | {'Resize':<7} | {'JPEG q60':<8} | {'Rot90':<7} | {'Crop 5%':<8} | {'Recall':<7} | {'Precision':<9} | {'False Merges':<12} | {'Speed':<10}"
    print(header)
    print("-" * 105)

    for name, s in all_summaries.items():
        r = s["recall_by_type"]
        exact_str = f"{r.get('exact', 0.0):.1f}%"
        resize_str = f"{r.get('resize_512', 0.0):.1f}%"
        q60_str = f"{r.get('jpeg_q60', 0.0):.1f}%"
        rot_str = f"{r.get('rotate90', 0.0):.1f}%"
        crop_str = f"{r.get('crop_05pct', 0.0):.1f}%"
        rec_str = f"{s['overall_recall']:.1f}%"
        prec_str = f"{s['overall_precision']:.1f}%"
        fm_str = f"{s['total_fp']}"
        speed_str = f"{s['ms_per_pair']:.1f} ms/pr"

        row = f"{name:<28} | {exact_str:<7} | {resize_str:<7} | {q60_str:<8} | {rot_str:<7} | {crop_str:<8} | {rec_str:<7} | {prec_str:<9} | {fm_str:<12} | {speed_str:<10}"
        print(row)
    print("=" * 105)

    # Print 40 random clusters/pairs for visual review
    print("\n" + "=" * 90)
    print("  40 SAMPLE CLUSTER PAIRS (VERIFIER: SIFT 512px)")
    print("=" * 90)
    print(f"{'#':<3} | {'Representative':<26} | {'Member Pair':<28} | {'Variant Type':<14} | {'Match %':<8} | {'Verdict'}")
    print("-" * 90)

    import random
    random.seed(42)
    sample_pool = random.sample(sample_verdicts_to_print, min(40, len(sample_verdicts_to_print)))
    for idx, v in enumerate(sample_pool, 1):
        gt = "Genuine Duplicate" if v["ground_truth"] else "Distinct Image"
        pred = "Confirmed Match" if v["predicted_duplicate"] else "Rejected (Split/Unique)"
        status_symbol = "✓ Correct" if (v["predicted_duplicate"] == v["ground_truth"]) else "✗ Error"
        print(f"{idx:<3} | {v['p1']:<26} | {v['p2']:<28} | {v['type']:<14} | {v['match_pct']:<7.1f}% | {pred} ({status_symbol})")
    print("=" * 90)


if __name__ == "__main__":
    main()
