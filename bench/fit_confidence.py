"""
bench/fit_confidence.py — Fit fused duplicate confidence model on the tuning set.

Features:
1. phash_distance (Hamming distance, 0..64)
2. best_clip_cosine (max dot product across 0°, 90°, 180°, 270°)
3. sift_inlier_ratio (inliers / good_matches, 0..1)
4. log_inlier_count (log(1 + inliers))

Rules:
- Exact MD5 matches are fixed at 100.0% confidence.
- Trained ONLY on the 60% tuning set (split.json, seed 42).
- Evaluated on the 40% held-out test set (reliability table in 10% buckets, Brier score).
- Saves model to models/duplicate_confidence.json.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sqlite3
import time
from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize

from core.embeddings import deserialize_embedding
from core.index import FlatIPIndex
from core.verify import KeypointVerifier, load_proxy_grayscale


def compute_hamming_distance(h1: str | None, h2: str | None) -> int:
    if not h1 or not h2 or len(h1) != len(h2):
        return 64
    try:
        val1 = int(h1, 16)
        val2 = int(h2, 16)
        return bin(val1 ^ val2).count("1")
    except ValueError:
        return 64


def get_best_clip_cosine(row1: sqlite3.Row, row2: sqlite3.Row) -> float:
    v1 = deserialize_embedding(row1["clip_vector"]).astype(np.float32)
    # Compare v1 against all rotations of row2
    sims = [float(np.dot(v1, deserialize_embedding(row2["clip_vector"]).astype(np.float32)))]
    for rot in ["clip_vector_90", "clip_vector_180", "clip_vector_270"]:
        if row2[rot] is not None:
            v2_rot = deserialize_embedding(row2[rot]).astype(np.float32)
            sims.append(float(np.dot(v1, v2_rot)))
    return max(sims)


def extract_features_parallel(
    image_rows: list[sqlite3.Row],
    verifier: KeypointVerifier,
    max_workers: int = 8,
) -> dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]]:
    """Extract SIFT keypoints and descriptors in parallel across image rows."""
    keypoint_dict = {}

    def _worker(row: sqlite3.Row):
        fp = row["file_path"]
        try:
            gray = load_proxy_grayscale(fp, proxy_size=verifier.proxy_size)
            kp, des = verifier.extract_features(gray)
            return row["id"], kp, des
        except Exception:
            return row["id"], [], None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img_id, kp, des in ex.map(_worker, image_rows):
            keypoint_dict[img_id] = (kp, des)

    return keypoint_dict


def build_pairs_and_extract_features(
    images: list[sqlite3.Row],
    keypoints_cache: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]],
    verifier: KeypointVerifier,
    gt_records: dict[str, dict],
    is_tuning: bool = True,
    sample_neg_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Construct pairs (true duplicates and negatives from FAISS/pHash/random)
    and compute feature vectors.
    """
    rng = random.Random(seed)
    id_to_row = {r["id"]: r for r in images}
    id_to_orig = {r["id"]: gt_records[Path(r["file_path"]).name]["original"] for r in images}
    id_to_trans = {r["id"]: gt_records[Path(r["file_path"]).name]["transform"] for r in images}

    # Group by original
    orig_to_ids: dict[str, list[int]] = {}
    for r in images:
        orig = id_to_orig[r["id"]]
        orig_to_ids.setdefault(orig, []).append(r["id"])

    pos_pair_set = set()
    neg_pair_set = set()

    # 1. Positive pairs: Every variant paired with its original
    for orig, members in orig_to_ids.items():
        # Find the original file
        orig_ids = [m for m in members if id_to_trans[m] == "original"]
        orig_id = orig_ids[0] if orig_ids else members[0]
        for m in members:
            if m != orig_id:
                p = (min(m, orig_id), max(m, orig_id))
                pos_pair_set.add(p)

    # 2. Hard negatives from FAISS top-K
    idx = FlatIPIndex(dim=512)
    ids = [r["id"] for r in images]
    vecs = [deserialize_embedding(r["clip_vector"]) for r in images]
    idx.add(ids, vecs)
    sims, nbrs = idx.search(np.vstack(vecs), k=20)

    for i, q_id in enumerate(ids):
        q_orig = id_to_orig[q_id]
        for rank in range(sims.shape[1]):
            nbr_id = int(nbrs[i, rank])
            if nbr_id == q_id or nbr_id == -1:
                continue
            nbr_orig = id_to_orig[nbr_id]
            pair = (min(q_id, nbr_id), max(q_id, nbr_id))
            if q_orig == nbr_orig:
                pos_pair_set.add(pair)
            else:
                neg_pair_set.add(pair)

    # 3. Candidate negatives from pHash (distance <= 14)
    # Sample subset if too many
    all_origs_list = list(orig_to_ids.keys())
    for _ in range(len(images) * 2):
        o1, o2 = rng.sample(all_origs_list, 2)
        id1 = rng.choice(orig_to_ids[o1])
        id2 = rng.choice(orig_to_ids[o2])
        pair = (min(id1, id2), max(id1, id2))
        r1, r2 = id_to_row[pair[0]], id_to_row[pair[1]]
        dist = compute_hamming_distance(r1["phash"], r2["phash"])
        if dist <= 14 or rng.random() < 0.15:
            neg_pair_set.add(pair)

    # Subsample negative pairs to match target ratio if needed
    neg_list = list(neg_pair_set)
    pos_list = list(pos_pair_set)

    # Exclude exact MD5 duplicates (as MD5 matches are fixed at 100%)
    filtered_pos = []
    for p in pos_list:
        r1, r2 = id_to_row[p[0]], id_to_row[p[1]]
        if r1["md5"] == r2["md5"]:
            continue
        filtered_pos.append(p)

    filtered_neg = []
    for p in neg_list:
        r1, r2 = id_to_row[p[0]], id_to_row[p[1]]
        if r1["md5"] == r2["md5"]:
            continue
        filtered_neg.append(p)

    rng.shuffle(filtered_neg)
    target_neg_count = int(len(filtered_pos) * sample_neg_ratio)
    selected_neg = filtered_neg[:target_neg_count] if len(filtered_neg) > target_neg_count else filtered_neg

    all_pairs = [(p, 1.0) for p in filtered_pos] + [(p, 0.0) for p in selected_neg]
    rng.shuffle(all_pairs)

    X_list = []
    y_list = []
    pair_meta = []

    for (id1, id2), label in all_pairs:
        r1 = id_to_row[id1]
        r2 = id_to_row[id2]

        phash_dist = float(compute_hamming_distance(r1["phash"], r2["phash"]))
        clip_cos = float(get_best_clip_cosine(r1, r2))

        kp1, des1 = keypoints_cache[id1]
        kp2, des2 = keypoints_cache[id2]

        match_res = verifier.match_features(kp1, des1, kp2, des2)
        inlier_ratio = float(match_res.inlier_ratio)
        log_inliers = float(np.log1p(match_res.inlier_count))

        feat = [phash_dist, clip_cos, inlier_ratio, log_inliers]
        X_list.append(feat)
        y_list.append(label)
        pair_meta.append({
            "id1": id1,
            "id2": id2,
            "path1": r1["file_path"],
            "path2": r2["file_path"],
            "label": label,
            "features": feat,
            "inlier_count": match_res.inlier_count,
        })

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), pair_meta


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Fit regularized logistic regression:
    logit = intercept + dot(X, weights)
    Returns (weights, intercept).
    """
    n_features = X.shape[1]

    def loss_and_grad(params: np.ndarray):
        bias = params[0]
        w = params[1:]
        logits = np.dot(X, w) + bias
        # Numerically stable logistic loss
        loss = np.sum(np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))) + 0.5 * l2_reg * np.sum(w**2)
        p = 1.0 / (1.0 + np.exp(-logits))
        grad_bias = np.sum(p - y)
        grad_w = np.dot(X.T, (p - y)) + l2_reg * w
        return loss, np.concatenate(([grad_bias], grad_w))

    init_params = np.zeros(n_features + 1, dtype=np.float64)
    res = minimize(loss_and_grad, init_params, jac=True, method="L-BFGS-B")
    if not res.success:
        print(f"Warning: optimization warning: {res.message}")

    intercept = float(res.x[0])
    weights = res.x[1:].astype(float)
    return weights, intercept


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """
    Compute reliability table in 10% buckets and Brier score.
    """
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    buckets = []
    for b in range(10):
        low = b * 0.1
        high = (b + 1) * 0.1
        if b == 9:
            mask = (y_prob >= low) & (y_prob <= high + 1e-6)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            obs_pos = float(np.mean(y_true[mask]))
            cal_err = float(abs(mean_pred - obs_pos))
        else:
            mean_pred = 0.0
            obs_pos = 0.0
            cal_err = 0.0

        buckets.append({
            "bucket_idx": b,
            "range": f"[{low:.1f}, {high:.1f})" if b < 9 else f"[{low:.1f}, 1.0]",
            "count": count,
            "mean_predicted": round(mean_pred, 4),
            "observed_positive_rate": round(obs_pos, 4),
            "calibration_error": round(cal_err, 4),
        })

    return {
        "brier_score": round(brier_score, 5),
        "total_pairs": len(y_true),
        "buckets": buckets,
    }


def main():
    print("=" * 70)
    print("  ARGUS FUSED DUPLICATE CONFIDENCE MODEL TRAINING & CALIBRATION")
    print("=" * 70)

    # Load split and ground truth
    split_path = Path("bench/split.json")
    split = json.loads(split_path.read_text())
    tuning_origs = set(split["tuning_originals"])
    test_origs = set(split["test_originals"])

    import csv
    gt_records = {}
    with open("data/benchmark/ground_truth.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_records[row["filename"]] = {
                "original": row["original_filename"],
                "transform": row["transformation"],
                "params": row["parameters"],
            }

    # Load images from eval_cache.db
    conn = sqlite3.connect("bench/eval_cache.db")
    conn.row_factory = sqlite3.Row
    all_images = conn.execute(
        """SELECT id, file_path, md5, phash, clip_vector, clip_vector_90, clip_vector_180, clip_vector_270
           FROM images ORDER BY id"""
    ).fetchall()

    def get_orig(fp):
        name = Path(fp).name
        return gt_records.get(name, {}).get("original", "")

    tuning_images = [r for r in all_images if get_orig(r["file_path"]) in tuning_origs]
    test_images = [r for r in all_images if get_orig(r["file_path"]) in test_origs]

    print(f"Loaded {len(tuning_images)} tuning images (120 originals) and {len(test_images)} test images (80 originals).")

    verifier = KeypointVerifier()

    # Step 1: Pre-extract keypoints in parallel
    print("\n[1/4] Pre-extracting SIFT keypoints on proxies in parallel...")
    t0 = time.perf_counter()
    kp_tuning = extract_features_parallel(tuning_images, verifier, max_workers=8)
    kp_test = extract_features_parallel(test_images, verifier, max_workers=8)
    t1 = time.perf_counter()
    print(f"      Keypoint extraction complete in {t1 - t0:.2f}s ({len(kp_tuning) + len(kp_test)} images).")

    # Step 2: Build training set from tuning split ONLY
    print("\n[2/4] Constructing candidate pairs from TUNING set...")
    t0 = time.perf_counter()
    X_train, y_train, train_meta = build_pairs_and_extract_features(
        tuning_images, kp_tuning, verifier, gt_records, is_tuning=True, sample_neg_ratio=1.0, seed=42
    )
    t1 = time.perf_counter()
    pos_train = int(np.sum(y_train == 1.0))
    neg_train = int(np.sum(y_train == 0.0))
    print(f"      Tuning pairs: {len(y_train)} total ({pos_train} positives, {neg_train} negatives) built in {t1 - t0:.2f}s.")

    # Step 3: Fit regularized logistic regression on TUNING set only
    print("\n[3/4] Fitting Logistic Regression on TUNING set...")
    weights, intercept = fit_logistic_regression(X_train, y_train, l2_reg=1.0)
    feature_names = ["phash_distance", "best_clip_cosine", "sift_inlier_ratio", "log_inlier_count"]
    print(f"      Fitted Intercept: {intercept:.4f}")
    for name, w in zip(feature_names, weights):
        print(f"      Weight for {name:<22}: {w:+.4f}")

    # Step 4: Evaluate calibration on HELD-OUT TEST SET
    print("\n[4/4] Evaluating calibration on HELD-OUT TEST set (80 originals)...")
    t0 = time.perf_counter()
    X_test, y_test, test_meta = build_pairs_and_extract_features(
        test_images, kp_test, verifier, gt_records, is_tuning=False, sample_neg_ratio=1.0, seed=42
    )
    t1 = time.perf_counter()
    pos_test = int(np.sum(y_test == 1.0))
    neg_test = int(np.sum(y_test == 0.0))
    print(f"      Test pairs: {len(y_test)} total ({pos_test} positives, {neg_test} negatives) built in {t1 - t0:.2f}s.")

    logits_test = np.dot(X_test, weights) + intercept
    probs_test = 1.0 / (1.0 + np.exp(-logits_test))

    calib = evaluate_calibration(y_test, probs_test)

    print("\n" + "=" * 75)
    print("  HELD-OUT TEST SET CALIBRATION & RELIABILITY TABLE (10% BUCKETS)")
    print("=" * 75)
    print(f"Brier Score: {calib['brier_score']:.5f} (lower is better, 0.0 = perfect calibration)")
    print(f"Total Evaluated Test Pairs: {calib['total_pairs']}")
    print("-" * 75)
    print(f"{'Probability Bucket':<20} | {'Count':<8} | {'Mean Pred %':<14} | {'Observed Pos %':<16} | {'Cal Error':<10}")
    print("-" * 75)
    for b in calib["buckets"]:
        cnt = b["count"]
        mp = f"{b['mean_predicted'] * 100:.1f}%" if cnt > 0 else "—"
        op = f"{b['observed_positive_rate'] * 100:.1f}%" if cnt > 0 else "—"
        ce = f"{b['calibration_error'] * 100:.2f}%" if cnt > 0 else "—"
        print(f"{b['range']:<20} | {cnt:<8} | {mp:<14} | {op:<16} | {ce:<10}")
    print("=" * 75 + "\n")

    # Save model as JSON
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "duplicate_confidence.json"

    model_data = {
        "model_type": "logistic_regression",
        "features": feature_names,
        "weights": [round(float(w), 6) for w in weights],
        "intercept": round(float(intercept), 6),
        "md5_fixed_confidence": 100.0,
        "training_metadata": {
            "dataset": "60% tuning split (seed 42, 120 originals)",
            "training_pairs": len(y_train),
            "positives": pos_train,
            "negatives": neg_train,
            "l2_regularization": 1.0,
        },
        "test_calibration": calib,
    }

    model_path.write_text(json.dumps(model_data, indent=2), encoding="utf-8")
    print(f"Model saved to: {model_path}\n")


if __name__ == "__main__":
    main()
