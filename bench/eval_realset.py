"""
bench/eval_realset.py — Comprehensive evaluation of Argus on the real-photo test set.

Evaluates:
1. Duplicates found vs missed per transformation (exact, screenshot PNG, WhatsApp, 1080p).
2. False merges (clusters containing images from different distinct originals).
3. Count of reference_uncertain clusters.
4. Deletion plans for conservative, balanced, and aggressive (files, bytes).
5. Clusters where the recommended keeper is not a full-resolution file.
6. HTML contact sheet for any false merges.
"""

from __future__ import annotations

import base64
import csv
import io
from pathlib import Path
import sqlite3
import sys
from typing import Any

from PIL import Image

from core.config import load_config
from core.keeper import choose_keeper
from core.ledger import plan


def format_bytes(b: int | float) -> str:
    if b < 1024:
        return f"{b} B"
    kb = b / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    mb = kb / 1024
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024
    return f"{gb:.2f} GB"


def make_thumb_b64(path: Path, max_sz: int = 160) -> str:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_sz, max_sz), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def run_evaluation(
    db_path: str | Path = "~/dedupe_test/realset.db",
    gt_path: str | Path = "~/dedupe_test/realset/photos/ground_truth.csv",
    out_dir: str | Path = "bench/results",
) -> None:
    db_file = Path(db_path).expanduser().resolve()
    gt_file = Path(gt_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not db_file.exists():
        print(f"Error: Database file not found: {db_file}", file=sys.stderr)
        sys.exit(1)
    if not gt_file.exists():
        print(f"Error: Ground truth CSV not found: {gt_file}", file=sys.stderr)
        sys.exit(1)

    # 1. Load ground truth
    # Rows: file, original, transformation
    gt_by_file: dict[str, dict[str, str]] = {}
    dups_by_orig: dict[str, list[dict[str, str]]] = {}
    originals: set[str] = set()

    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file"]
            gt_by_file[fname] = row
            orig = row["original"]
            trans = row["transformation"]
            if trans == "original":
                originals.add(fname)
            else:
                dups_by_orig.setdefault(orig, []).append(row)

    print("====================================================================")
    print("  ARGUS Real-Photo Benchmark Evaluation vs Ground Truth")
    print("====================================================================")
    print(f"Database     : {db_file}")
    print(f"Ground Truth : {gt_file} ({len(gt_by_file)} files, {len(originals)} originals)")

    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Query all images
    img_rows = conn.execute(
        """
        SELECT id, cluster_id, file_path, file_size, width, height, format, md5,
               is_keeper, match_pct, resolution_pct, confidence, link_signal,
               quality_status, category_top1, category_top1_prob
        FROM images
        """
    ).fetchall()

    db_images_by_fname: dict[str, dict] = {}
    for r in img_rows:
        fn = Path(r["file_path"]).name
        db_images_by_fname[fn] = dict(r)

    # Group images by cluster
    clusters_members: dict[int, list[dict]] = {}
    for fn, img in db_images_by_fname.items():
        cid = img["cluster_id"]
        if cid is not None:
            clusters_members.setdefault(cid, []).append(img)

    print(f"Total images in DB: {len(db_images_by_fname)}")
    print(f"Total clusters   : {len(clusters_members)}")
    print(f"Clustered images : {sum(len(m) for m in clusters_members.values())}")
    print(f"Unclustered images: {len(db_images_by_fname) - sum(len(m) for m in clusters_members.values())}")
    print("--------------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # Metric 1: Duplicate Retrieval vs Ground Truth
    # ---------------------------------------------------------
    trans_stats: dict[str, dict[str, int]] = {
        "exact_byte_copy": {"expected": 0, "found": 0, "missed": 0},
        "screenshot_png_1x": {"expected": 0, "found": 0, "missed": 0},
        "whatsapp_1600_q75": {"expected": 0, "found": 0, "missed": 0},
        "resize_1080_q70": {"expected": 0, "found": 0, "missed": 0},
    }

    missed_duplicates: list[dict[str, Any]] = []

    for orig_fn, dups in dups_by_orig.items():
        orig_img = db_images_by_fname.get(orig_fn)
        orig_cid = orig_img["cluster_id"] if orig_img else None

        for d in dups:
            dup_fn = d["file"]
            trans = d["transformation"]
            if trans not in trans_stats:
                continue
            trans_stats[trans]["expected"] += 1

            dup_img = db_images_by_fname.get(dup_fn)
            dup_cid = dup_img["cluster_id"] if dup_img else None

            # A duplicate is correctly found if it shares the same cluster as its original
            if orig_cid is not None and dup_cid == orig_cid:
                trans_stats[trans]["found"] += 1
            else:
                trans_stats[trans]["missed"] += 1
                missed_duplicates.append({
                    "file": dup_fn,
                    "original": orig_fn,
                    "transformation": trans,
                    "dup_cluster_id": dup_cid,
                    "orig_cluster_id": orig_cid,
                })

    print("====================================================================")
    print("1. Duplicate Detection by Transformation")
    print("====================================================================")
    header = f"{'Transformation':<20} | {'Expected':<8} | {'Found':<8} | {'Missed':<8} | {'Recall':<8}"
    print(header)
    print("-" * 60)
    total_exp = 0
    total_found = 0
    total_missed = 0
    for t_name, s in trans_stats.items():
        rec = (s["found"] / s["expected"] * 100) if s["expected"] else 0.0
        print(f"{t_name:<20} | {s['expected']:>8} | {s['found']:>8} | {s['missed']:>8} | {rec:>7.1f}%")
        total_exp += s["expected"]
        total_found += s["found"]
        total_missed += s["missed"]
    print("-" * 60)
    total_rec = (total_found / total_exp * 100) if total_exp else 0.0
    print(f"{'OVERALL':<20} | {total_exp:>8} | {total_found:>8} | {total_missed:>8} | {total_rec:>7.1f}%")
    print("====================================================================\n")

    if missed_duplicates:
        print(f"Missed Duplicates ({len(missed_duplicates)} files):")
        for m in missed_duplicates:
            print(f"  - {m['file']} (trans: {m['transformation']}, orig: {m['original']}, dup_cid={m['dup_cluster_id']}, orig_cid={m['orig_cluster_id']})")
        print()

    # ---------------------------------------------------------
    # Metric 2: False Merges (Clusters containing multiple originals)
    # ---------------------------------------------------------
    false_merge_clusters: list[dict[str, Any]] = []

    for cid, members in sorted(clusters_members.items()):
        member_origs = set()
        for m in members:
            fn = Path(m["file_path"]).name
            gt = gt_by_file.get(fn)
            if gt:
                member_origs.add(gt["original"])

        if len(member_origs) > 1:
            false_merge_clusters.append({
                "cluster_id": cid,
                "distinct_originals": sorted(member_origs),
                "member_count": len(members),
                "members": members,
            })

    print("====================================================================")
    print("2. False Merges (Clusters with > 1 Distinct Original)")
    print("====================================================================")
    if false_merge_clusters:
        print(f"WARNING: Found {len(false_merge_clusters)} cluster(s) with false merges:")
        for fmc in false_merge_clusters:
            print(f"  Cluster #{fmc['cluster_id']} ({fmc['member_count']} files, {len(fmc['distinct_originals'])} originals: {', '.join(fmc['distinct_originals'])}):")
            for m in fmc["members"]:
                fn = Path(m["file_path"]).name
                gt_orig = gt_by_file.get(fn, {}).get("original", "unknown")
                gt_trans = gt_by_file.get(fn, {}).get("transformation", "unknown")
                print(f"    - {fn:<30} (gt_orig: {gt_orig}, trans: {gt_trans}, match_pct: {m['match_pct']}%, conf: {m['confidence']})")
    else:
        print(f"SUCCESS: 0 false merges detected! All {len(clusters_members)} clusters contain only one distinct original.")
    print("====================================================================\n")

    # ---------------------------------------------------------
    # Metric 3: Count of reference_uncertain clusters
    # ---------------------------------------------------------
    c_rows = conn.execute("SELECT id, representative_id, reference_uncertain FROM clusters").fetchall()
    ref_unc_clusters = [r["id"] for r in c_rows if r["reference_uncertain"]]

    print("====================================================================")
    print("3. Reference Uncertain Clusters")
    print("====================================================================")
    print(f"Total clusters                  : {len(c_rows)}")
    print(f"Reference uncertain clusters    : {len(ref_unc_clusters)} ({(len(ref_unc_clusters)/len(c_rows)*100 if c_rows else 0):.1f}%)")
    if ref_unc_clusters:
        print(f"Cluster IDs: {ref_unc_clusters}")
    print("====================================================================\n")

    # ---------------------------------------------------------
    # Metric 4: Deletion Plans Across Modes
    # ---------------------------------------------------------
    # Run ledger.plan on a temporary copy of db so the real DB stays untouched
    import shutil
    import tempfile
    plan_results: dict[str, Any] = {}
    modes = ["conservative", "balanced", "aggressive"]

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        shutil.copy2(db_file, tmp_db.name)
        tmp_conn = sqlite3.connect(tmp_db.name)
        tmp_conn.row_factory = sqlite3.Row
        for m in modes:
            summary = plan(tmp_conn, mode=m)
            plan_results[m] = summary
        tmp_conn.close()

    total_photo_bytes = sum(m["file_size"] for m in db_images_by_fname.values())

    print("====================================================================")
    print("4. Safe Deletion Plans by Mode (Simulated via ledger.plan)")
    print("====================================================================")
    header = f"{'Mode':<14} | {'Files to Delete':<16} | {'Bytes Freed':<14} | {'% of Total Size':<16}"
    print(header)
    print("-" * 65)
    for m in modes:
        s = plan_results[m]
        pct = (s.bytes_freed / total_photo_bytes * 100) if total_photo_bytes else 0.0
        print(f"{m:<14} | {s.file_count:>16} | {format_bytes(s.bytes_freed):>14} | {pct:>15.1f}%")
    print("====================================================================\n")

    # ---------------------------------------------------------
    # Metric 5: Recommended Keeper Resolution Check
    # ---------------------------------------------------------
    non_fullres_keepers: list[dict[str, Any]] = []

    for cid, members in sorted(clusters_members.items()):
        max_res = max(m["width"] * m["height"] for m in members)

        for m in modes:
            keeper_res = choose_keeper(members, mode=m)
            k_id = keeper_res["keeper_id"]
            k_m = next((x for x in members if x["id"] == k_id), None)
            if k_m:
                k_res = k_m["width"] * k_m["height"]
                if k_res < max_res:
                    non_fullres_keepers.append({
                        "cluster_id": cid,
                        "mode": m,
                        "keeper_file": Path(k_m["file_path"]).name,
                        "keeper_dims": f"{k_m['width']}x{k_m['height']}",
                        "keeper_size": format_bytes(k_m["file_size"]),
                        "max_dims": f"{max(x['width'] for x in members)}x{max(x['height'] for x in members)}",
                        "reason": keeper_res["reasons"].get(k_id, "N/A"),
                        "swap_proposal": keeper_res.get("swap_proposal"),
                    })

    print("====================================================================")
    print("5. Clusters Where Recommended Keeper is NOT Full Resolution")
    print("====================================================================")
    if non_fullres_keepers:
        print(f"Found {len(non_fullres_keepers)} occurrence(s):")
        for item in non_fullres_keepers:
            print(f"  Cluster #{item['cluster_id']} [{item['mode']}]: Keeper {item['keeper_file']} ({item['keeper_dims']}, {item['keeper_size']}) < Max ({item['max_dims']})")
            print(f"    Reason: {item['reason']}")
            if item["swap_proposal"]:
                print(f"    Swap proposal: {item['swap_proposal']}")
    else:
        print("NONE: In all modes and clusters, the recommended keeper is a full-resolution file!")
    print("====================================================================\n")

    # ---------------------------------------------------------
    # Step 5: Contact Sheet of False Merges
    # ---------------------------------------------------------
    cs_path = out_path / "false_merges_contact_sheet.html"
    realset_cs_path = Path("~/dedupe_test/realset/false_merges_contact_sheet.html").expanduser().resolve()

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Argus Real-Photo Benchmark — False Merges Contact Sheet</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }",
        "    h1 { margin-bottom: 8px; font-size: 1.5rem; }",
        "    .subtitle { color: #94a3b8; margin-bottom: 24px; font-size: 0.9rem; }",
        "    .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 24px; }",
        "    .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; color: #38bdf8; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }",
        "    .item { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; font-size: 0.8rem; }",
        "    .item img { width: 100%; height: 140px; object-fit: contain; background: #000; border-radius: 4px; margin-bottom: 8px; }",
        "    .item-title { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }",
        "    .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin-top: 4px; }",
        "    .badge-orig { background: #0369a1; color: #e0f2fe; }",
        "    .badge-trans { background: #4338ca; color: #e0e7ff; }",
        "    .empty-state { padding: 48px; text-align: center; color: #34d399; font-size: 1.2rem; font-weight: 600; background: #064e3b; border: 1px solid #059669; border-radius: 8px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Argus Real-Photo Benchmark: False Merges Contact Sheet</h1>",
        f"  <div class='subtitle'>Dataset: ROxford5k subset (300 originals + 160 duplicates) &bull; Clusters Evaluated: {len(clusters_members)} &bull; False Merges: {len(false_merge_clusters)}</div>",
    ]

    if not false_merge_clusters:
        html_parts.append(
            "  <div class='empty-state'>✓ Perfect Specificity: 0 False Merges Detected.<br>"
            "  <span style='font-size:0.9rem; font-weight:400; color:#a7f3d0; margin-top:8px; display:inline-block;'>"
            "  Every single cluster generated by Argus contains only copies of a single distinct original photograph.</span></div>"
        )
    else:
        for fmc in false_merge_clusters:
            html_parts.append(f"  <div class='card'>")
            html_parts.append(f"    <div class='card-title'>Cluster #{fmc['cluster_id']} &bull; {fmc['member_count']} members &bull; {len(fmc['distinct_originals'])} distinct originals: {', '.join(fmc['distinct_originals'])}</div>")
            html_parts.append(f"    <div class='grid'>")
            for m in fmc["members"]:
                p = Path(m["file_path"])
                fn = p.name
                gt_orig = gt_by_file.get(fn, {}).get("original", "unknown")
                gt_trans = gt_by_file.get(fn, {}).get("transformation", "unknown")
                thumb_src = make_thumb_b64(p)
                html_parts.append(f"      <div class='item'>")
                html_parts.append(f"        <img src='{thumb_src}' alt='{fn}'>")
                html_parts.append(f"        <div class='item-title' title='{fn}'>{fn}</div>")
                html_parts.append(f"        <div><span class='badge badge-orig'>Orig: {gt_orig}</span></div>")
                html_parts.append(f"        <div><span class='badge badge-trans'>{gt_trans}</span></div>")
                html_parts.append(f"        <div style='color:#94a3b8; margin-top:4px;'>{m['width']}x{m['height']} &bull; {format_bytes(m['file_size'])}</div>")
                html_parts.append(f"        <div style='color:#94a3b8;'>Match: {m['match_pct']}% &bull; Res: {m['resolution_pct']}%</div>")
                html_parts.append(f"      </div>")
            html_parts.append(f"    </div>")
            html_parts.append(f"  </div>")

    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)

    for p in (cs_path, realset_cs_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html_content)

    print("====================================================================")
    print("5. Contact Sheet Generated")
    print("====================================================================")
    print(f"Saved to: {cs_path}")
    print(f"Also at : {realset_cs_path}")
    print("====================================================================")

    conn.close()


if __name__ == "__main__":
    run_evaluation()
