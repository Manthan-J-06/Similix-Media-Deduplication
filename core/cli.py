"""
core/cli.py — Command-line interface for Argus.

Usage:
    python -m core.cli scan <folder>
    argus scan <folder>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from core.config import load_config, DEFAULT_CONFIG_PATH
from core.stage0 import run_stage0
from core.stage1 import run_stage1
from db.connection import init_db
from sources.local_folder import LocalFolderSource


def _render_progress_bar(current: int, total: int, prefix: str = "", bar_length: int = 30) -> None:
    """Render a clean zero-dependency terminal progress bar."""
    if total <= 0:
        percent = 100.0
        filled = bar_length
    else:
        percent = min(100.0, (current / total) * 100.0)
        filled = int(bar_length * current // total)

    bar = "=" * filled + (">" if filled < bar_length else "") + " " * (bar_length - filled - (1 if filled < bar_length else 0))
    sys.stdout.write(f"\r{prefix:<25} [{bar}] {current}/{total} ({percent:.1f}%)")
    sys.stdout.flush()
    if current >= total and total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def cmd_scan(args: argparse.Namespace) -> None:
    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Target directory does not exist: {folder}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    print(f"==================================================")
    print(f"  ARGUS Image Deduplication Pipeline")
    print(f"==================================================")
    print(f"Target folder : {folder}")
    print(f"Database      : {db_path}")
    print(f"Batch size    : {cfg.get('general', {}).get('batch_size', 500)}")
    print()

    start_time = time.time()
    source = LocalFolderSource(folder)

    # ── Phase 1 & 2: Stage 0 (Scan, MD5, RAW pairs, Exact duplicates) ──
    print("[1/2] Running Stage 0: File discovery, MD5 hashing & exact matching...")
    stage0_res = run_stage0(source, conn, cfg)
    print(f"      Discovered / updated : {stage0_res.files_discovered}")
    print(f"      MD5 hashes computed  : {stage0_res.files_hashed}")
    print(f"      RAW+JPEG pairs marked: {stage0_res.raw_pairs_marked}")
    print(f"      Exact clusters found : {stage0_res.clusters_created} ({stage0_res.files_clustered} files)")
    print()

    # ── Phase 3 & 4: Stage 1 (pHash on thumbnails, Near-duplicate clustering) ──
    print("[2/2] Running Stage 1: Perceptual hashing (draft mode) & near duplicates...")
    stage1_res = run_stage1(source, conn, cfg)
    print(f"      pHashes computed     : {stage1_res.hashes_computed}")
    print(f"      Near clusters created: {stage1_res.clusters_created}")
    print(f"      Added to clusters    : {stage1_res.members_added_to_existing}")
    print(f"      Total near-clustered : {stage1_res.files_clustered}")
    print()

    # ── Summary ──
    elapsed = time.time() - start_time
    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    total_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    clustered_images = conn.execute("SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL").fetchone()[0]
    keepers = conn.execute("SELECT COUNT(*) FROM images WHERE is_keeper = 1").fetchone()[0]
    duplicates_to_review = clustered_images - keepers

    print("--------------------------------------------------")
    print(f"Scan complete in {elapsed:.2f}s")
    print(f"Total images indexed   : {total_images}")
    print(f"Total duplicate groups : {total_clusters}")
    print(f"Files in clusters      : {clustered_images}")
    print(f"Recommended keepers    : {keepers}")
    print(f"Redundant copies found : {duplicates_to_review}")
    print("--------------------------------------------------")


def cmd_embed(args: argparse.Namespace) -> None:
    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Target directory does not exist: {folder}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    from core.embeddings import run_embeddings, get_device
    device = get_device()
    stage2_cfg = cfg.get("stage2", {})
    rotations = stage2_cfg.get("enable_rotation_embeddings", False)

    print("==================================================")
    print("  ARGUS Stage 2 CLIP Embedding Pipeline")
    print("==================================================")
    print(f"Target folder : {folder}")
    print(f"Database      : {db_path}")
    print(f"Model         : {stage2_cfg.get('clip_model', 'ViT-B-32')}")
    print(f"Weights       : {stage2_cfg.get('clip_pretrained', 'laion2b_s34b_b79k')}")
    print(f"Device        : {device} ({'Apple Silicon GPU' if device.type == 'mps' else 'CPU'})")
    print(f"Batch size    : {stage2_cfg.get('batch_size', 32)}")
    print(f"Rotations     : {'Enabled (0°, 90°, 180°, 270°)' if rotations else 'Disabled (0° only)'}")
    print()

    source = LocalFolderSource(folder)
    start_time = time.time()

    def on_progress(current: int, total: int):
        _render_progress_bar(current, total, prefix="Embedding thumbnails")

    res = run_embeddings(source, conn, cfg, progress_callback=on_progress)

    elapsed = time.time() - start_time
    print()
    print("--------------------------------------------------")
    print(f"Embedding completed in {elapsed:.2f}s")
    print(f"Newly embedded images : {res.images_embedded}")
    print(f"Previously embedded   : {res.images_skipped}")
    print(f"Device used           : {res.device_used}")
    print("--------------------------------------------------")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Argus: Local-first Image Deduplication System",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan a directory for duplicate images")
    scan_parser.add_argument("folder", help="Path to image folder")
    scan_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    scan_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    embed_parser = subparsers.add_parser("embed", help="Compute CLIP embeddings for images")
    embed_parser.add_argument("folder", help="Path to image folder")
    embed_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    embed_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    args = parser.parse_args(argv)
    if args.subcommand == "scan":
        cmd_scan(args)
    elif args.subcommand == "embed":
        cmd_embed(args)


if __name__ == "__main__":
    main()
