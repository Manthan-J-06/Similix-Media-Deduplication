"""
bench/eval_split.py — Evaluation script for realset tuning and test splits.
"""

import csv
import json
from pathlib import Path
import sqlite3

def load_ground_truth(csv_path: Path):
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["file"]] = {
                "original": row["original"],
                "transformation": row["transformation"],
            }
    return gt

def eval_db(db_path: Path, split_name: str, gt: dict, expected_origs: set):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 1. Total files and clusters
    images = conn.execute("SELECT id, file_path, cluster_id, match_pct, resolution_pct FROM images").fetchall()
    clusters = conn.execute("SELECT id, representative_id, reference_uncertain, member_count FROM clusters").fetchall()
    
    file_map = {}
    for img in images:
        name = Path(img["file_path"]).name
        file_map[name] = img

    # Group by cluster_id
    cluster_members = {}
    unclustered = []
    for name, img in file_map.items():
        cid = img["cluster_id"]
        if cid is not None:
            cluster_members.setdefault(cid, []).append((name, img))
        else:
            unclustered.append((name, img))

    # Evaluate duplicates recall
    # True duplicate: transformation != 'original'
    dupes_total = 0
    dupes_found = 0
    missed_dupes = []

    for name, info in gt.items():
        if info["original"] in expected_origs and info["transformation"] != "original":
            dupes_total += 1
            orig_name = info["original"]
            img = file_map.get(name)
            orig_img = file_map.get(orig_name)
            if img and orig_img and img["cluster_id"] is not None and img["cluster_id"] == orig_img["cluster_id"]:
                dupes_found += 1
            else:
                missed_dupes.append((name, orig_name, img["cluster_id"] if img else None))

    # Evaluate False Merges
    false_merge_clusters = {}
    for cid, members in cluster_members.items():
        origs = set()
        for name, _ in members:
            orig = gt.get(name, {}).get("original", name)
            origs.add(orig)
        if len(origs) > 1:
            false_merge_clusters[cid] = {
                "originals": sorted(list(origs)),
                "members": [name for name, _ in members],
            }

    # Uncertain clusters
    ref_uncertain = conn.execute("SELECT count(*) FROM clusters WHERE reference_uncertain = 1").fetchone()[0]

    conn.close()

    recall = (dupes_found / dupes_total * 100.0) if dupes_total > 0 else 100.0
    print(f"=== {split_name.upper()} RESULTS ({db_path}) ===")
    print(f"Total files in DB: {len(file_map)}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Duplicates recall: {dupes_found}/{dupes_total} ({recall:.2f}%)")
    if missed_dupes:
        print(f"Missed duplicates ({len(missed_dupes)}): {missed_dupes}")
    print(f"False merge clusters: {len(false_merge_clusters)}")
    for cid, info in false_merge_clusters.items():
        print(f"  Cluster {cid}: originals={info['originals']}, members={info['members']}")
    print(f"reference_uncertain clusters: {ref_uncertain}")
    print("==============================================\n")

    return {
        "recall": recall,
        "dupes_found": dupes_found,
        "dupes_total": dupes_total,
        "false_merges": len(false_merge_clusters),
        "false_merge_details": false_merge_clusters,
        "reference_uncertain": ref_uncertain,
    }

if __name__ == "__main__":
    split_info = json.loads(Path("bench/realset_split.json").read_text(encoding="utf-8"))
    gt = load_ground_truth(Path("/Users/manthanj/dedupe_test/realset/photos/ground_truth.csv"))

    tuning_origs = set(split_info["tuning_originals"])
    test_origs = set(split_info["test_originals"])

    eval_db(Path("/Users/manthanj/dedupe_test/tuning.db"), "tuning", gt, tuning_origs)
    eval_db(Path("/Users/manthanj/dedupe_test/test.db"), "test", gt, test_origs)
