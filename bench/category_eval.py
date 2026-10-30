"""
bench/category_eval.py — Evaluate zero-shot semantic categorization accuracy.

Features:
- Takes a CSV of file,label (ground truth).
- Reports overall top-1 accuracy and accuracy per label.
- Reports number and percentage of images flagged as 'uncertain'.
- Generates and prints a detailed confusion matrix table.
- Provides a helper command (--draft) to generate a draft CSV with the model's
  current predictions for easy manual verification and correction.

Usage:
    # 1. Generate draft CSV for manual labeling
    python -m bench.category_eval --draft draft_labels.csv --db argus.db

    # 2. Evaluate against ground truth CSV
    python -m bench.category_eval --csv ground_truth_categories.csv --db argus.db
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

from core.config import load_config, DEFAULT_CONFIG_PATH
from db.connection import init_db


def generate_draft_csv(conn: sqlite3.Connection, output_path: str | Path, limit: int | None = None) -> int:
    """
    Generate a draft CSV containing (file, label) with model predictions.
    Users can inspect and correct this draft by hand.
    """
    query = "SELECT file_path, category_top1 FROM images WHERE category_top1 IS NOT NULL ORDER BY id"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    rows = conn.execute(query).fetchall()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        for r in rows:
            fname = Path(r["file_path"]).name
            pred = r["category_top1"] or "uncertain"
            writer.writerow([fname, pred])

    return len(rows)


def evaluate_categories(
    csv_path: str | Path,
    conn: sqlite3.Connection,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Evaluate category predictions against a ground truth CSV (file, label).

    Returns:
        Dict with overall accuracy, per-label accuracy, uncertain counts, and confusion matrix.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    # Load ground truth CSV
    ground_truth: list[tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Check header
        file_col = None
        label_col = None
        for col in (reader.fieldnames or []):
            if col.lower() in ("file", "filename", "file_path", "image"):
                file_col = col
            elif col.lower() in ("label", "category", "ground_truth", "class"):
                label_col = col

        if not file_col or not label_col:
            raise ValueError(f"CSV must contain 'file' and 'label' headers. Found: {reader.fieldnames}")

        for row in reader:
            f_val = str(row[file_col]).strip()
            l_val = str(row[label_col]).strip().lower()
            if f_val and l_val:
                ground_truth.append((f_val, l_val))

    if not ground_truth:
        raise ValueError("Ground truth CSV is empty.")

    # Load database predictions indexed by basename and full path
    db_rows = conn.execute(
        "SELECT id, file_path, category_top1, category_top1_prob FROM images"
    ).fetchall()

    db_by_name: dict[str, dict[str, Any]] = {}
    db_by_path: dict[str, dict[str, Any]] = {}
    for r in db_rows:
        p = Path(r["file_path"])
        entry = {
            "id": r["id"],
            "file_path": r["file_path"],
            "pred": (r["category_top1"] or "unlabeled").lower(),
            "prob": r["category_top1_prob"],
        }
        db_by_name[p.name] = entry
        db_by_path[str(p)] = entry

    total = 0
    correct = 0
    uncertain_count = 0
    per_label_total = Counter()
    per_label_correct = Counter()
    confusion = defaultdict(lambda: Counter())

    all_true_labels = set()
    all_pred_labels = set()

    missing_in_db = []

    for file_spec, true_label in ground_truth:
        # Match by filename or path
        f_name = Path(file_spec).name
        matched = db_by_name.get(f_name) or db_by_path.get(file_spec)

        if not matched:
            missing_in_db.append(file_spec)
            continue

        pred_label = matched["pred"]
        total += 1
        all_true_labels.add(true_label)
        all_pred_labels.add(pred_label)

        per_label_total[true_label] += 1
        confusion[true_label][pred_label] += 1

        if pred_label in ("uncertain", "needs label"):
            uncertain_count += 1

        if pred_label == true_label:
            correct += 1
            per_label_correct[true_label] += 1

    overall_acc = (correct / total * 100.0) if total > 0 else 0.0
    uncertain_pct = (uncertain_count / total * 100.0) if total > 0 else 0.0

    per_label_acc = {}
    for l in sorted(all_true_labels):
        t_cnt = per_label_total[l]
        c_cnt = per_label_correct[l]
        per_label_acc[l] = {
            "total": t_cnt,
            "correct": c_cnt,
            "accuracy": (c_cnt / t_cnt * 100.0) if t_cnt > 0 else 0.0,
        }

    return {
        "total_evaluated": total,
        "correct": correct,
        "overall_accuracy": overall_acc,
        "uncertain_count": uncertain_count,
        "uncertain_percent": uncertain_pct,
        "per_label": per_label_acc,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "all_true_labels": sorted(all_true_labels),
        "all_pred_labels": sorted(all_pred_labels),
        "missing_in_db": missing_in_db,
    }


def print_evaluation_report(results: dict[str, Any]) -> None:
    """Print a clean, formatted evaluation report including confusion matrix."""
    total = results["total_evaluated"]
    correct = results["correct"]
    acc = results["overall_accuracy"]
    unc = results["uncertain_count"]
    unc_pct = results["uncertain_percent"]

    print("====================================================================================================")
    print("  ARGUS Zero-Shot Semantic Categorization Evaluation Report")
    print("====================================================================================================")
    print(f"Total Samples Evaluated : {total}")
    print(f"Correct Predictions     : {correct}")
    print(f"Top-1 Overall Accuracy  : {acc:.2f}%")
    print(f"Uncertain Predictions   : {unc} ({unc_pct:.2f}%)")
    if results.get("missing_in_db"):
        print(f"Missing in Database     : {len(results['missing_in_db'])}")
    print("----------------------------------------------------------------------------------------------------")
    print(f"{'Label':<18} | {'Total':<8} | {'Correct':<8} | {'Accuracy (%)'}")
    print("-" * 55)
    for l, stats in results["per_label"].items():
        print(f"{l:<18} | {stats['total']:<8} | {stats['correct']:<8} | {stats['accuracy']:>10.2f}%")
    print("----------------------------------------------------------------------------------------------------")

    # Confusion Matrix
    true_labels = results["all_true_labels"]
    pred_labels = results["all_pred_labels"]
    # Ensure all true labels are included in cols plus 'uncertain' if present
    cols = sorted(list(set(pred_labels) | set(true_labels)))
    if "uncertain" in cols:
        cols.remove("uncertain")
        cols.append("uncertain")

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Model Predictions):")
    header = f"{'True / Pred':<16} | " + " | ".join(f"{c[:10]:>10}" for c in cols)
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for r_label in true_labels:
        row_counts = [results["confusion"].get(r_label, {}).get(c, 0) for c in cols]
        row_str = f"{r_label:<16} | " + " | ".join(f"{cnt:>10}" for cnt in row_counts)
        print(row_str)

    print("-" * len(header))
    print("====================================================================================================\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bench/category_eval.py",
        description="Argus Semantic Categorization Evaluation and Draft Generation",
    )
    parser.add_argument("--csv", default=None, help="Path to ground truth CSV (file, label)")
    parser.add_argument("--draft", default=None, help="Generate a draft CSV with model predictions to manual verify")
    parser.add_argument("--db", default=None, help="Path to SQLite database file")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of draft samples")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    if args.draft:
        count = generate_draft_csv(conn, args.draft, limit=args.limit)
        print(f"Draft CSV created with {count} image predictions: {args.draft}")
        print("You can now review or edit the labels in this CSV and run with --csv to evaluate.")
        return

    if args.csv:
        res = evaluate_categories(args.csv, conn, cfg)
        print_evaluation_report(res)
    else:
        print("Specify either --draft <output.csv> or --csv <ground_truth.csv>.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
