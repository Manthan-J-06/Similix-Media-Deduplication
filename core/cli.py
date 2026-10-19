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
    exts_display = ", ".join(sorted(source.extensions))
    print(f"Accepted exts : {exts_display} (case-insensitive: .JPG, .HEIC, .jpeg, .webp, .tiff)")
    print()

    # ── Phase 1 & 2: Stage 0 (Scan, MD5, RAW pairs, Exact duplicates) ──
    print("[1/2] Running Stage 0: File discovery, MD5 hashing & exact matching...")
    stage0_res = run_stage0(source, conn, cfg)
    print(f"      Discovered / updated : {stage0_res.files_discovered}")
    if stage0_res.files_unchanged > 0:
        print(f"      Unchanged (in DB)    : {stage0_res.files_unchanged}")
    print(f"      MD5 hashes computed  : {stage0_res.files_hashed}")
    print(f"      RAW+JPEG pairs marked: {stage0_res.raw_pairs_marked}")
    print(f"      Exact clusters found : {stage0_res.clusters_created} ({stage0_res.files_clustered} files)")
    if stage0_res.files_skipped > 0:
        print(f"      Skipped / unreadable : {stage0_res.files_skipped}")
        for path_str, reason in stage0_res.skipped_reasons:
            print(f"        - {Path(path_str).name}: {reason}")
    print()

    # ── Phase 3 & 4: Stage 1 (pHash on thumbnails, Near-duplicate clustering) ──
    print("[2/2] Running Stage 1: Perceptual hashing (draft mode) & near duplicates...")
    stage1_res = run_stage1(source, conn, cfg)
    print(f"      pHashes computed     : {stage1_res.hashes_computed}")
    print(f"      Near clusters created: {stage1_res.clusters_created}")
    print(f"      Added to clusters    : {stage1_res.members_added_to_existing}")
    print(f"      Total near-clustered : {stage1_res.files_clustered}")
    if stage1_res.files_skipped > 0:
        print(f"      Skipped in Stage 1   : {stage1_res.files_skipped}")
        for path_str, reason in stage1_res.skipped_reasons:
            print(f"        - {Path(path_str).name}: {reason}")
    print()

    # ── Summary ──
    elapsed = time.time() - start_time
    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    total_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    clustered_images = conn.execute("SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL").fetchone()[0]
    keepers = conn.execute("SELECT COUNT(*) FROM images WHERE is_keeper = 1").fetchone()[0]
    duplicates_to_review = clustered_images - keepers
    total_skipped = stage0_res.files_skipped + stage1_res.files_skipped
    all_skipped = stage0_res.skipped_reasons + stage1_res.skipped_reasons

    print("--------------------------------------------------")
    print(f"Scan complete in {elapsed:.2f}s")
    print(f"Total images indexed   : {total_images}")
    print(f"Total duplicate groups : {total_clusters}")
    print(f"Files in clusters      : {clustered_images}")
    print(f"Cluster anchors        : {keepers}")
    print(f"Redundant copies found : {duplicates_to_review}")
    print(f"Skipped: {total_skipped}")
    if total_skipped > 0:
        for path_str, reason in all_skipped:
            print(f"  - {Path(path_str).name}: {reason}")
    print("--------------------------------------------------")


def cmd_embed(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    source = None
    if args.folder:
        folder = Path(args.folder).resolve()
        if not folder.exists() or not folder.is_dir():
            print(f"Error: Target directory does not exist: {folder}", file=sys.stderr)
            sys.exit(1)
        source = LocalFolderSource(folder)
        target_display = str(folder)
    else:
        has_gdrive = conn.execute("SELECT 1 FROM images WHERE source = 'gdrive' LIMIT 1").fetchone() is not None
        if has_gdrive:
            from sources.google_drive import GoogleDriveSource
            source = GoogleDriveSource(conn=conn, stage="stage2")
            target_display = "Google Drive"
        else:
            print("Error: Target directory required when database has no Google Drive images", file=sys.stderr)
            sys.exit(1)

    from core.embeddings import run_embeddings, get_device
    device = get_device()
    stage2_cfg = cfg.get("stage2", {})
    rotations = stage2_cfg.get("enable_rotation_embeddings", False)

    print("==================================================")
    print("  ARGUS Stage 2 CLIP Embedding Pipeline")
    print("==================================================")
    print(f"Target        : {target_display}")
    print(f"Database      : {db_path}")
    print(f"Model         : {stage2_cfg.get('clip_model', 'ViT-B-32')}")
    print(f"Weights       : {stage2_cfg.get('clip_pretrained', 'laion2b_s34b_b79k')}")
    print(f"Device        : {device} ({'Apple Silicon GPU' if device.type == 'mps' else 'CPU'})")
    print(f"Batch size    : {stage2_cfg.get('batch_size', 32)}")
    print(f"Rotations     : {'Enabled (0°, 90°, 180°, 270°)' if rotations else 'Disabled (0° only)'}")
    print()

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

    if has_gdrive:
        from sources.google_drive import print_bandwidth_summary
        print()
        print_bandwidth_summary(conn)


def cmd_cluster(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    stage2_cfg = cfg.get("stage2", {})
    index_path = args.index or stage2_cfg.get("faiss_index_path", str(Path(db_path).with_suffix(".faiss")))

    print("==================================================")
    print("  ARGUS Stage 2 Duplicate Clustering")
    print("==================================================")
    print(f"Database         : {db_path}")
    print(f"FAISS Index      : {index_path}")
    print(f"Cosine threshold : {stage2_cfg.get('cosine_threshold', 0.92)}")
    print(f"Neighbors (K)    : {stage2_cfg.get('faiss_k', 20)}")
    print()

    start_time = time.time()
    from core.cluster import run_stage2_clustering
    res = run_stage2_clustering(conn, cfg, index_path=index_path)
    elapsed = time.time() - start_time

    # Summary
    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    total_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    clustered_images = conn.execute("SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL").fetchone()[0]
    keepers = conn.execute("SELECT COUNT(*) FROM images WHERE is_keeper = 1").fetchone()[0]
    signal_counts = dict(
        conn.execute("SELECT link_signal, COUNT(*) FROM images WHERE cluster_id IS NOT NULL GROUP BY link_signal").fetchall()
    )

    print("--------------------------------------------------")
    print(f"Clustering complete in {elapsed:.2f}s")
    print(f"Candidate pairs found       : {res.candidates_found}")
    print(f"New Stage 2 clusters        : {res.clusters_created}")
    print(f"Added to existing clusters  : {res.members_added_to_existing}")
    print(f"Clusters merged             : {res.clusters_merged}")
    print()
    print(f"Total images in library     : {total_images}")
    print(f"Total duplicate groups      : {total_clusters}")
    print(f"Files clustered             : {clustered_images}")
    print(f"Cluster anchors             : {keepers}")
    print(f"Redundant copies            : {clustered_images - keepers}")
    print(f"Signal breakdown            : {signal_counts}")
    print("--------------------------------------------------")


def cmd_verify(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    stage3_cfg = cfg.get("stage3", {})
    algo = stage3_cfg.get("algorithm", "sift")
    min_inl = stage3_cfg.get("min_inliers", 20)
    min_rat = stage3_cfg.get("min_inlier_ratio", 0.65)

    print("==================================================")
    print("  ARGUS Stage 3 Geometric Verification")
    print("==================================================")
    print(f"Database         : {db_path}")
    print(f"Algorithm        : {algo.upper()}")
    print(f"Min Inliers      : {min_inl}")
    print(f"Min Inlier Ratio : {min_rat}")
    print()

    has_gdrive = conn.execute("SELECT 1 FROM images WHERE source = 'gdrive' LIMIT 1").fetchone() is not None
    source = None
    if args.folder:
        folder = Path(args.folder).resolve()
        if folder.exists() and folder.is_dir():
            source = LocalFolderSource(folder)
    elif has_gdrive:
        from sources.google_drive import GoogleDriveSource
        source = GoogleDriveSource(conn=conn, stage="stage3")

    from core.verify import run_stage3_verification
    res = run_stage3_verification(conn, cfg, source=source)

    total_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    clustered_images = conn.execute("SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL").fetchone()[0]

    print("--------------------------------------------------")
    print(f"Verification complete in {res.elapsed_seconds:.2f}s ({res.time_per_1000:.2f}s per 1,000 images)")
    print(f"Pairs evaluated          : {res.pairs_evaluated}")
    print(f"Pairs verified           : {res.pairs_verified}")
    print(f"Impostor members split   : {res.members_split}")
    print(f"Clusters disbanded       : {res.clusters_disbanded}")
    print(f"Remaining duplicate groups: {total_clusters}")
    print(f"Remaining clustered files: {clustered_images}")
    print("--------------------------------------------------")

    if has_gdrive:
        from sources.google_drive import print_bandwidth_summary
        print()
        print_bandwidth_summary(conn)


def cmd_quality(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    q_cfg = cfg.get("quality", {})
    proxy_sz = q_cfg.get("proxy_size", 1024)
    min_inl = q_cfg.get("min_align_inliers", 15)
    min_rat = q_cfg.get("min_align_inlier_ratio", 0.50)

    print("==================================================")
    print("  ARGUS Quality Scoring & Geometric Alignment")
    print("==================================================")
    print(f"Database         : {db_path}")
    print(f"Proxy Size       : {proxy_sz}px")
    print(f"Min Inliers      : {min_inl}")
    print(f"Min Inlier Ratio : {min_rat}")
    print()

    has_gdrive = conn.execute("SELECT 1 FROM images WHERE source = 'gdrive' LIMIT 1").fetchone() is not None
    source = None
    if args.folder:
        folder = Path(args.folder).resolve()
        if folder.exists() and folder.is_dir():
            source = LocalFolderSource(folder)
    elif has_gdrive:
        from sources.google_drive import GoogleDriveSource
        source = GoogleDriveSource(conn=conn, stage="stage4")

    from core.quality import run_quality_pipeline
    res = run_quality_pipeline(conn, cfg, source=source)

    print("--------------------------------------------------")
    print(f"Quality evaluation complete in {res.elapsed_seconds:.2f}s")
    print(f"Clusters evaluated       : {res.clusters_evaluated}")
    print(f"Members evaluated        : {res.members_evaluated}")
    print(f"Members aligned          : {res.members_aligned}")
    print(f"Members unalignable      : {res.members_unalignable}")
    print("--------------------------------------------------")

    if has_gdrive:
        from sources.google_drive import print_bandwidth_summary
        print()
        print_bandwidth_summary(conn)


def cmd_keeper(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    mode = args.mode or cfg.get("keeper", {}).get("default_mode", "balanced")

    print("==================================================")
    print("  ARGUS Stage 4 Keeper Selection & Deletion Plan")
    print("==================================================")
    print(f"Database         : {db_path}")
    print(f"Keeper Mode      : {mode.upper()}")
    print()

    from core.keeper import run_keeper_pipeline
    res = run_keeper_pipeline(
        conn,
        mode=mode,
        config=cfg,
        allow_resolution_loss=args.allow_resolution_loss,
    )

    print("--------------------------------------------------")
    print(f"Clusters evaluated       : {res.clusters_evaluated}")
    print(f"Recommended keepers      : {res.total_keepers}")
    print(f"Delete-safe members      : {res.total_delete_safe}")
    print(f"Members for manual review: {res.total_review}")
    print(f"Total potential savings  : {res.total_bytes_saved:,} bytes ({res.total_bytes_saved / (1024*1024):.2f} MB)")
    print("--------------------------------------------------")


def cmd_plan(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    mode = args.mode or cfg.get("keeper", {}).get("default_mode", "balanced")

    print("==================================================")
    print("  ARGUS Deletion Planning (core/ledger)")
    print("==================================================")
    print(f"Database         : {db_path}")
    print(f"Mode             : {mode.upper()}")
    print()

    from core.ledger import plan
    summary = plan(conn, mode=mode, config=cfg, trash_dir=args.trash_dir)

    print("--------------------------------------------------")
    print(f"Batch ID                 : {summary.batch_id}")
    print(f"Files planned for trash  : {summary.file_count}")
    print(f"Total space to reclaim   : {summary.bytes_freed:,} bytes ({summary.bytes_freed / (1024*1024):.2f} MB)")
    print(f"Clusters with deletions  : {len(summary.per_cluster_keeper)}")
    print(f"Trash target             : {summary.trash_dir}/{summary.batch_id}/")
    print("--------------------------------------------------")
    if summary.file_count > 0:
        print(f"To execute this batch and move files to trash, run:")
        print(f"  argus apply {summary.batch_id} --confirm")
    else:
        print("No files eligible for deletion in this mode.")


def cmd_apply(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    if args.dry_run:
        from core.ledger import dry_run
        print(f"Running dry-run simulation for batch '{args.batch_id}'...")
        try:
            d_res = dry_run(conn, args.batch_id)
        except ValueError as e:
            if f"Batch '{args.batch_id}' not found" in str(e):
                print(f"Batch '{args.batch_id}' not found; run `argus history`", file=sys.stderr)
                sys.exit(1)
            raise
        print(f"Batch ID       : {d_res.batch_id}")
        print(f"Total planned  : {d_res.total_planned}")
        print(f"Ready to move  : {d_res.ready_count}")
        print(f"Refused / skip : {d_res.refused_count}")
        for ref in d_res.refusals:
            print(f"  - Refused {ref['path']}: {ref['reason']}")
        return

    if not args.confirm:
        print("Error: The 'apply' command moves files to trash and requires the --confirm flag.", file=sys.stderr)
        print(f"Run: argus apply {args.batch_id} --confirm", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print("  ARGUS Applying Deletion Batch (core/ledger)")
    print("==================================================")
    print(f"Batch ID         : {args.batch_id}")
    print(f"Database         : {db_path}")
    print()

    from core.ledger import execute
    try:
        res = execute(conn, batch_id=args.batch_id, confirm=True)
    except ValueError as e:
        if f"Batch '{args.batch_id}' not found" in str(e):
            print(f"Batch '{args.batch_id}' not found; run `argus history`", file=sys.stderr)
            sys.exit(1)
        raise

    print("--------------------------------------------------")
    print(f"Total planned    : {res.total_planned}")
    print(f"Successfully moved: {res.moved_count}")
    print(f"Refused (safety) : {res.refused_count}")
    print(f"Failed moves     : {res.failed_count}")
    print(f"Space reclaimed  : {res.bytes_moved:,} bytes ({res.bytes_moved / (1024*1024):.2f} MB)")
    print("--------------------------------------------------")
    if res.refusals:
        print(f"Refusals ({len(res.refusals)}):")
        for r in res.refusals[:10]:
            print(f"  - {r['path']}: {r['reason']}")
        if len(res.refusals) > 10:
            print(f"  ... and {len(res.refusals) - 10} more")
    if res.failures:
        print(f"Failures ({len(res.failures)}):")
        for f in res.failures:
            print(f"  - {f['path']}: {f['error']}")
    if res.moved_count > 0:
        print(f"\nTo rollback this batch, run:\n  argus undo {args.batch_id}")


def cmd_undo(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    print("==================================================")
    print("  ARGUS Undo / Restore Trashed Files")
    print("==================================================")
    print(f"Database         : {db_path}")

    from core.ledger import undo, undo_file

    if args.file:
        print(f"Restoring file   : {args.file}")
        ok, msg = undo_file(conn, original_path=args.file)
        if ok:
            print(f"Success: {msg}")
        else:
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.batch_id:
        print("Error: Specify a batch_id or --file <path> to undo.", file=sys.stderr)
        sys.exit(1)

    print(f"Batch ID         : {args.batch_id}")
    print()

    try:
        res = undo(conn, batch_id=args.batch_id)
    except ValueError as e:
        if f"Batch '{args.batch_id}' not found" in str(e):
            print(f"Batch '{args.batch_id}' not found; run `argus history`", file=sys.stderr)
            sys.exit(1)
        raise

    print("--------------------------------------------------")
    print(f"Files restored   : {res.restored_count}")
    print(f"Conflicts        : {res.conflict_count}")
    print("--------------------------------------------------")
    if res.conflicts:
        print("Restore Conflicts (files were left in trash):")
        for p, reason in res.conflicts:
            print(f"  - {p}: {reason}")


def cmd_purge(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)
    min_age = args.min_age_days if args.min_age_days is not None else cfg.get("ledger", {}).get("purge_min_age_days", 30)

    if not args.confirm:
        print("Error: The 'purge' command permanently removes files from trash and requires the --confirm flag.", file=sys.stderr)
        print(f"Run: argus purge {args.batch_id} --confirm", file=sys.stderr)
        sys.exit(1)

    from core.ledger import purge
    try:
        res = purge(conn, batch_id=args.batch_id, confirm=True, max_age_days=min_age)
    except ValueError as e:
        if f"Batch '{args.batch_id}' not found" in str(e):
            print(f"Batch '{args.batch_id}' not found; run `argus history`", file=sys.stderr)
            sys.exit(1)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print("  ARGUS Purge Trashed Files")
    print("==================================================")
    print(f"Batch ID         : {res.batch_id}")
    print(f"Purged files     : {res.purged_count}")
    print(f"Bytes purged     : {res.bytes_purged:,}")
    print("--------------------------------------------------")


def cmd_history(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    from core.ledger import get_history
    batches = get_history(conn, limit=args.limit)

    print("====================================================================================================")
    print(f"  ARGUS Deletion Ledger History ({len(batches)} batches)")
    print("====================================================================================================")
    header = f"{'Batch ID':<36} | {'Created At':<20} | {'Mode':<12} | {'Status':<10} | {'Files':<6} | {'Freed (MB)'}"
    print(header)
    print("-" * 100)
    for b in batches:
        created = (b.get("created_at") or "")[:19]
        freed_mb = (b.get("bytes_freed") or 0) / (1024 * 1024)
        print(f"{b['batch_id']:<36} | {created:<20} | {b.get('mode',''):<12} | {b.get('status',''):<10} | {b.get('file_count',0):<6} | {freed_mb:>9.2f}")
    print("=" * 100)


def cmd_categorize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db
    if not db_path and getattr(args, "folder", None):
        target = Path(args.folder).expanduser().resolve()
        if target.is_file() and target.suffix == ".db":
            db_path = str(target)
        elif target.with_suffix(".db").is_file():
            db_path = str(target.with_suffix(".db"))
        elif (target / "argus.db").is_file():
            db_path = str(target / "argus.db")
        elif (target.parent / f"{target.name}.db").is_file():
            db_path = str(target.parent / f"{target.name}.db")
        else:
            db_path = str(target)
    if not db_path:
        db_path = cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    from core.categories import run_categorization

    print("==================================================")
    print("  ARGUS Zero-Shot Semantic Categorization")
    print("==================================================")
    print(f"Database : {db_path}")
    print(f"Mode     : {'Force re-categorize' if args.force else 'Resumable (skip labeled)'}")
    print()

    res = run_categorization(
        conn,
        cfg,
        force=args.force,
        progress_callback=lambda c, t: _render_progress_bar(c, t, "Categorizing"),
    )

    print("\n--------------------------------------------------")
    print(f"Images categorized : {res.images_categorized}")
    print(f"Images skipped     : {res.images_skipped}")
    print(f"Clusters updated   : {res.clusters_updated}")
    print(f"Flagged uncertain  : {res.uncertain_count}")
    print("--------------------------------------------------")
    if res.category_counts:
        print("Category Distribution:")
        for cat, count in sorted(res.category_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / (res.images_categorized or 1)) * 100
            print(f"  {cat:<18} : {count:>5} ({pct:>5.1f}%)")
    print("==================================================")


def cmd_search(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    from core.categories import search_images_by_text

    results = search_images_by_text(args.query, conn, cfg, limit=args.limit)

    print("====================================================================================================")
    print(f"  ARGUS Semantic Text Search: \"{args.query}\" ({len(results)} results)")
    print("====================================================================================================")
    header = f"{'Rank':<4} | {'Score':<7} | {'Category':<14} | {'Prob':<6} | {'Resolution':<11} | {'File Path'}"
    print(header)
    print("-" * 100)
    for i, r in enumerate(results, 1):
        res_str = f"{r.get('width') or 0}x{r.get('height') or 0}"
        cat = r.get("category_top1") or "unlabeled"
        prob = f"{r.get('category_top1_prob'):.2f}" if r.get("category_top1_prob") is not None else "N/A"
        print(f"{i:<4} | {r['score']:>6.4f} | {cat:<14} | {prob:<6} | {res_str:<11} | {r['file_path']}")
    print("=" * 100)


def cmd_list(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    params: list[Any] = []
    sql = """
        SELECT id, file_path, file_size, width, height, format,
               category_top1, category_top1_prob, cluster_id, is_keeper
        FROM images
    """
    if args.category:
        sql += " WHERE LOWER(category_top1) = LOWER(?)"
        params.append(args.category)
    sql += " ORDER BY id LIMIT ?"
    params.append(args.limit)

    rows = conn.execute(sql, params).fetchall()

    print("====================================================================================================")
    cat_title = f" [Category: {args.category}]" if args.category else ""
    print(f"  ARGUS Image List{cat_title} ({len(rows)} images displayed, limit {args.limit})")
    print("====================================================================================================")
    header = f"{'ID':<6} | {'Category':<14} | {'Prob':<6} | {'Resolution':<11} | {'Size (KB)':<10} | {'File Path'}"
    print(header)
    print("-" * 100)
    for r in rows:
        d = dict(r)
        res_str = f"{d.get('width') or 0}x{d.get('height') or 0}"
        cat = d.get("category_top1") or "unlabeled"
        prob = f"{d.get('category_top1_prob'):.2f}" if d.get('category_top1_prob') is not None else "N/A"
        sz_kb = f"{(d.get('file_size') or 0) / 1024:.1f}"
        print(f"{d['id']:<6} | {cat:<14} | {prob:<6} | {res_str:<11} | {sz_kb:<10} | {d['file_path']}")
    print("=" * 100)


def cmd_show(args: argparse.Namespace) -> None:
    from core.keeper import choose_keeper

    cfg = load_config(args.config) if getattr(args, "config", None) else load_config()
    mode = getattr(args, "mode", "balanced") or "balanced"
    db_path = args.db
    if not db_path and getattr(args, "folder", None):
        target = Path(args.folder).expanduser().resolve()
        if target.is_file() and target.suffix == ".db":
            db_path = str(target)
        elif target.with_suffix(".db").is_file():
            db_path = str(target.with_suffix(".db"))
        elif (target / "argus.db").is_file():
            db_path = str(target / "argus.db")
        elif (target / "demo.db").is_file():
            db_path = str(target / "demo.db")
        elif (target.parent / f"{target.name}.db").is_file():
            db_path = str(target.parent / f"{target.name}.db")
        else:
            db_path = str(target)
    if not db_path:
        db_path = cfg.get("general", {}).get("db_path", "argus.db")
    conn = init_db(db_path)

    clusters = conn.execute(
        """
        SELECT c.id, c.representative_id, c.reference_uncertain, c.stage_detected, c.member_count,
               r.file_path AS rep_path
        FROM clusters c
        LEFT JOIN images r ON c.representative_id = r.id
        ORDER BY c.id
        """
    ).fetchall()

    if not clusters:
        print(f"No clusters found in database: {db_path}")
        return

    print("=" * 115)
    print(f"  ARGUS Duplicate Clusters Summary: {db_path} ({len(clusters)} clusters) [Mode: {mode.upper()}]")
    print("=" * 115)

    total_swap_proposals = 0
    total_swap_bytes = 0

    for c in clusters:
        cid = c["id"]
        members = conn.execute(
            """
            SELECT id, cluster_id, file_path, file_size, width, height, format, md5,
                   match_pct, resolution_pct, confidence, quality_status,
                   ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio,
                   source_likelihood, jpeg_quality_est, is_low_confidence
            FROM images
            WHERE cluster_id = ?
            ORDER BY id ASC
            """,
            (cid,),
        ).fetchall()

        if not members:
            continue

        ref_unc = c["reference_uncertain"] if "reference_uncertain" in c.keys() else None
        decision = choose_keeper(
            members,
            mode=mode,
            config=cfg,
            reference_uncertain=ref_unc,
        )
        keeper_id = decision["keeper_id"]
        del_safe_ids = set(decision["delete_safe_ids"])
        review_ids = set(decision["review_ids"])

        annotated_members = []
        for m in members:
            d = dict(m)
            d["is_recommended_keeper"] = 1 if d["id"] == keeper_id else 0
            d["delete_safe"] = 1 if d["id"] in del_safe_ids else 0
            d["review"] = 1 if d["id"] in review_ids else 0
            annotated_members.append(d)

        annotated_members.sort(key=lambda x: (not x["is_recommended_keeper"], not x["delete_safe"], x["id"]))

        print(f"\nCluster {cid} ({len(members)} members):")
        header = f"  {'Rec Keeper':<10} | {'Del Safe':<8} | {'Review':<6} | {'Match %':<8} | {'Res %':<8} | {'Confidence':<10} | {'Size':<10} | {'File'}"
        print(header)
        print("  " + "-" * 110)

        for d in annotated_members:
            keeper_str = "YES" if d["is_recommended_keeper"] else "no"
            del_str = "YES" if d["delete_safe"] else "no"
            rev_str = "YES" if d["review"] else "no"
            match_str = f"{d['match_pct']:.1f}%" if d.get("match_pct") is not None else "N/A"
            res_str = f"{d['resolution_pct']:.1f}%" if d.get("resolution_pct") is not None else "N/A"
            conf = d.get("confidence")
            conf_str = f"{conf:.4f}" if conf is not None else "N/A"
            sz_kb = f"{(d.get('file_size') or 0) / 1024:.1f} KB"
            fname = Path(d["file_path"]).name

            print(f"  {keeper_str:<10} | {del_str:<8} | {rev_str:<6} | {match_str:>8} | {res_str:>8} | {conf_str:>10} | {sz_kb:>10} | {fname}")

        if decision.get("swap_proposal"):
            sp = decision["swap_proposal"]
            total_swap_proposals += 1
            total_swap_bytes += sp["bytes_saved"]
            from_name = Path(next((m["file_path"] for m in members if m["id"] == sp["from_id"]), str(sp["from_id"]))).name
            to_name = Path(next((m["file_path"] for m in members if m["id"] == sp["to_id"]), str(sp["to_id"]))).name
            print(f"  ⚡ Swap Proposal: Replace reference ({from_name}) with smaller keeper ({to_name})")
            print(f"     Saves: {sp['bytes_saved'] / 1024:.1f} KB ({sp['saving_pct']}%) | Match: {sp['match_pct']:.1f}% | Res: {sp['res_pct']:.1f}%")

    print("\n" + "=" * 115)
    print(f"  Summary: {len(clusters)} clusters | Mode: {mode.upper()} | Swap proposals: {total_swap_proposals} ({total_swap_bytes / 1024:.1f} KB at stake / {total_swap_bytes:,} bytes)")
    print("=" * 115)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from api.server import create_app

    cfg = load_config(args.config) if getattr(args, "config", None) else load_config()
    db_path = getattr(args, "db", None)
    if not db_path and getattr(args, "folder", None):
        target = Path(args.folder).expanduser().resolve()
        if target.is_file() and target.suffix == ".db":
            db_path = str(target)
        elif target.with_suffix(".db").is_file():
            db_path = str(target.with_suffix(".db"))
        elif (target / "argus.db").is_file():
            db_path = str(target / "argus.db")
        elif (target / "demo.db").is_file():
            db_path = str(target / "demo.db")
        elif (target.parent / f"{target.name}.db").is_file():
            db_path = str(target.parent / f"{target.name}.db")
        else:
            db_path = str(target)
    if not db_path:
        db_path = cfg.get("general", {}).get("db_path", "argus.db")

    resolved_db = str(Path(db_path).expanduser().resolve())

    app = create_app(
        db_path=resolved_db,
        trash_dir=getattr(args, "trash_dir", None),
        thumb_cache_dir=getattr(args, "thumb_cache_dir", None),
        config=cfg,
    )
    # Strictly bind to 127.0.0.1 only
    host = "127.0.0.1"
    port = getattr(args, "port", 8000) or 8000
    print(f"Starting Argus API server on http://{host}:{port} (db: {resolved_db})")
    uvicorn.run(app, host=host, port=port)


def cmd_drive_auth(args: argparse.Namespace) -> None:
    from sources.google_drive import (
        get_drive_credentials,
        DEFAULT_CLIENT_SECRET_PATH,
        DEFAULT_TOKEN_PATH,
    )

    client_secret = args.client_secret or DEFAULT_CLIENT_SECRET_PATH
    token_path = args.token or DEFAULT_TOKEN_PATH

    print("==================================================")
    print("  ARGUS Google Drive Authorization (READ-ONLY)")
    print("==================================================")
    print(f"Client Secret : {client_secret}")
    print(f"Token Path    : {token_path}")
    print("Scope         : https://www.googleapis.com/auth/drive.readonly")
    print()

    try:
        get_drive_credentials(
            client_secret_path=client_secret,
            token_path=token_path,
            prompt_if_missing=True,
        )
        print("Google Drive authorization successful!")
        print(f"Token stored securely at {token_path} with 0600 permissions.")
    except Exception as exc:
        print(f"Authorization error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_scan_drive(args: argparse.Namespace) -> None:
    from sources.google_drive import GoogleDriveSource, print_bandwidth_summary

    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "drive.db")
    conn = init_db(db_path)

    folder_id = args.folder
    all_drive = args.all
    if not folder_id and not all_drive:
        print("Error: Either --folder <id> or --all must be specified for scan-drive", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print("  ARGUS Google Drive Deduplication Pipeline")
    print("==================================================")
    target_str = "Whole Drive (--all)" if all_drive else f"Folder ID: {folder_id}"
    print(f"Target        : {target_str}")
    print(f"Database      : {db_path}")
    print(f"Permissions   : drive.readonly (READ-ONLY)")
    print()

    try:
        source = GoogleDriveSource(
            folder_id=folder_id,
            all_drive=all_drive,
            conn=conn,
            concurrency=getattr(args, "concurrency", 4),
            stage="stage0",
        )
    except Exception as exc:
        print(f"Error initializing Google Drive client: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Phase 1: Stage 0 (Metadata only, Zero downloads) ──
    print("[1/2] Running Stage 0: Metadata discovery & exact matching (0 downloads)...")
    stage0_res = run_stage0(source, conn, cfg)
    print(f"      Discovered / updated : {stage0_res.files_discovered}")
    if stage0_res.files_unchanged > 0:
        print(f"      Unchanged (in DB)    : {stage0_res.files_unchanged}")
    print(f"      MD5 hashes verified  : {stage0_res.files_hashed} (from Drive metadata)")
    print(f"      Exact clusters found : {stage0_res.clusters_created} ({stage0_res.files_clustered} files)")
    if stage0_res.files_skipped > 0:
        print(f"      Skipped (missing md5): {stage0_res.files_skipped}")
        for path_str, reason in stage0_res.skipped_reasons:
            print(f"        - {path_str}: {reason}")
    print()

    # ── Phase 2: Stage 1 (pHash on thumbnails) ──
    print("[2/2] Running Stage 1: Perceptual hashing via cached thumbnails...")
    source.stage = "stage1"
    stage1_res = run_stage1(source, conn, cfg)
    print(f"      pHashes computed     : {stage1_res.hashes_computed}")
    print(f"      Near clusters created: {stage1_res.clusters_created}")
    print(f"      Added to clusters    : {stage1_res.members_added_to_existing}")
    print(f"      Total near-clustered : {stage1_res.files_clustered}")
    if stage1_res.files_skipped > 0:
        print(f"      Skipped in Stage 1   : {stage1_res.files_skipped}")
        for path_str, reason in stage1_res.skipped_reasons:
            print(f"        - {path_str}: {reason}")
    print()

    # ── Bandwidth Accounting Summary ──
    print_bandwidth_summary(conn)


def cmd_confirm_drive(args: argparse.Namespace) -> None:
    from sources.google_drive import run_confirm_drive

    cfg = load_config(args.config)
    db_path = args.db or cfg.get("general", {}).get("db_path", "drive.db")
    conn = init_db(db_path)
    mode = getattr(args, "mode", "balanced")

    print("==================================================")
    print("  ARGUS Google Drive Full-Resolution Confirmation")
    print("==================================================")
    print(f"Database      : {db_path}")
    print(f"Mode          : {mode}")
    print(f"Permissions   : drive.readonly (READ-ONLY)")
    print()

    res = run_confirm_drive(conn, config=cfg, mode=mode)
    print("Confirmation Results:")
    print(f"  Members evaluated / confirmed: {len(res.confirmed_members)}")
    for m in res.confirmed_members:
        thumb_str = f"{m.thumb_match_pct:.1f}%" if m.thumb_match_pct is not None else "N/A"
        full_str = f"{m.full_match_pct:.1f}%" if m.full_match_pct is not None else "N/A"
        status_str = "delete_safe" if m.is_delete_safe else f"review ({m.reason})"
        print(f"    - Member {m.member_id} ({m.file_path}): thumb={thumb_str} -> full={full_str} [{status_str}]")
    print()
    print("Bandwidth Summary:")
    print(f"  Confirmation downloaded : {res.total_confirmed_bytes:,} bytes")
    print(f"  Total library size      : {res.total_library_bytes:,} bytes")
    print(f"  Bandwidth ratio         : {res.bandwidth_ratio_pct:.2f}%")
    print()


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
    embed_parser.add_argument("folder", nargs="?", default=None, help="Path to image folder (optional)")
    embed_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    embed_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    cluster_parser = subparsers.add_parser("cluster", help="Cluster duplicate images using FAISS index")
    cluster_parser.add_argument("folder", nargs="?", default=None, help="Path to image folder (optional)")
    cluster_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    cluster_parser.add_argument("--db", default=None, help="Path to SQLite database file")
    cluster_parser.add_argument("--index", default=None, help="Path to FAISS index file")

    verify_parser = subparsers.add_parser("verify", help="Verify clusters with geometric keypoint matching (Stage 3)")
    verify_parser.add_argument("folder", nargs="?", default=None, help="Path to image folder (optional)")
    verify_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    verify_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    quality_parser = subparsers.add_parser("quality", help="Score quality and align cluster proxies (Stage 4 prep)")
    quality_parser.add_argument("folder", nargs="?", default=None, help="Path to image folder (optional)")
    quality_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    quality_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    keeper_parser = subparsers.add_parser("keeper", help="Select keepers and calculate safe deletions (Stage 4)")
    keeper_parser.add_argument("--mode", choices=["conservative", "balanced", "aggressive"], default=None, help="Keeper policy mode")
    keeper_parser.add_argument("--allow-resolution-loss", action=argparse.BooleanOptionalAction, default=None, help="Allow keeper with lower pixel count than reference")
    keeper_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    keeper_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    plan_parser = subparsers.add_parser("plan", help="Plan a safe deletion batch into trash (Stage 4)")
    plan_parser.add_argument("--mode", choices=["conservative", "balanced", "aggressive"], default=None, help="Keeper policy mode")
    plan_parser.add_argument("--trash-dir", default=None, help="Custom trash directory path")
    plan_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    plan_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    apply_parser = subparsers.add_parser("apply", help="Execute a planned deletion batch (moves files to trash)")
    apply_parser.add_argument("batch_id", help="Batch ID to execute")
    apply_parser.add_argument("--confirm", action="store_true", help="Explicit confirmation required to move files")
    apply_parser.add_argument("--dry-run", action="store_true", help="Simulate execution without moving any files")
    apply_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    apply_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    undo_parser = subparsers.add_parser("undo", help="Restore files moved to trash")
    undo_parser.add_argument("batch_id", nargs="?", default=None, help="Batch ID to undo")
    undo_parser.add_argument("--file", default=None, help="Restore an individual file path")
    undo_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    undo_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    hist_parser = subparsers.add_parser("history", help="Show deletion batch history from ledger")
    hist_parser.add_argument("--limit", type=int, default=50, help="Maximum number of batches to display")
    hist_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    hist_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    purge_parser = subparsers.add_parser("purge", help="Permanently delete trashed files older than threshold")
    purge_parser.add_argument("batch_id", help="Batch ID to purge")
    purge_parser.add_argument("--confirm", action="store_true", help="Explicit confirmation required to permanently delete")
    purge_parser.add_argument("--min-age-days", type=int, default=None, help="Minimum age in days of batch to allow purging")
    purge_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    purge_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    cat_parser = subparsers.add_parser("categorize", help="Categorize images into zero-shot semantic categories")
    cat_parser.add_argument("folder", nargs="?", default=None, help="Path to image folder or database (optional)")
    cat_parser.add_argument("--db", default=None, help="Path to SQLite database file")
    cat_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    cat_parser.add_argument("--force", action="store_true", help="Re-categorize images already labeled")

    search_parser = subparsers.add_parser("search", help="Search images by natural language text query")
    search_parser.add_argument("query", help="Text search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to display")
    search_parser.add_argument("--db", default=None, help="Path to SQLite database file")
    search_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")

    list_parser = subparsers.add_parser("list", help="List images filtered by category")
    list_parser.add_argument("--category", "-c", default=None, help="Filter images by category (e.g. landscape, document, uncertain)")
    list_parser.add_argument("--limit", type=int, default=100, help="Maximum number of results to display")
    list_parser.add_argument("--db", default=None, help="Path to SQLite database file")
    list_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")

    show_parser = subparsers.add_parser("show", help="Show duplicate cluster members with match_pct, resolution_pct, and confidence")
    show_parser.add_argument("folder", nargs="?", default=None, help="Path to image folder or database (optional)")
    show_parser.add_argument(
        "--mode",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Keeper selection mode (default: balanced)",
    )
    show_parser.add_argument("--db", default=None, help="Path to SQLite database file")
    show_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")

    serve_parser = subparsers.add_parser("serve", help="Start the Argus FastAPI server on 127.0.0.1:8000")
    serve_parser.add_argument("folder", nargs="?", default=None, help="Path to image folder or database (optional)")
    serve_parser.add_argument("--db", default=None, help="Path to SQLite database file")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_parser.add_argument("--trash-dir", default=None, help="Path to trash directory")
    serve_parser.add_argument("--thumb-cache-dir", default=None, help="Path to thumbnail cache directory")
    serve_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")

    drive_auth_parser = subparsers.add_parser("drive-auth", help="Authorize Google Drive read-only access (OAuth)")
    drive_auth_parser.add_argument("--client-secret", default=None, help="Path to OAuth client_secret.json")
    drive_auth_parser.add_argument("--token", default=None, help="Path to drive_token.json")

    scan_drive_parser = subparsers.add_parser("scan-drive", help="Scan Google Drive for duplicate images")
    scan_drive_parser.add_argument("--folder", default=None, help="Google Drive folder ID to scan")
    scan_drive_parser.add_argument("--all", action="store_true", help="Scan entire Google Drive")
    scan_drive_parser.add_argument("--concurrency", type=int, default=4, help="Concurrent worker count (default: 4)")
    scan_drive_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    scan_drive_parser.add_argument("--db", default=None, help="Path to SQLite database file")

    confirm_drive_parser = subparsers.add_parser(
        "confirm-drive",
        help="Confirm candidate delete_safe Drive images at full resolution",
    )
    confirm_drive_parser.add_argument("--db", default=None, help="Path to SQLite database file")
    confirm_drive_parser.add_argument(
        "--mode",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Keeper selection mode (default: balanced)",
    )
    confirm_drive_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")

    args = parser.parse_args(argv)
    if args.subcommand == "scan":
        cmd_scan(args)
    elif args.subcommand == "embed":
        cmd_embed(args)
    elif args.subcommand == "cluster":
        cmd_cluster(args)
    elif args.subcommand == "verify":
        cmd_verify(args)
    elif args.subcommand == "quality":
        cmd_quality(args)
    elif args.subcommand == "keeper":
        cmd_keeper(args)
    elif args.subcommand == "plan":
        cmd_plan(args)
    elif args.subcommand == "apply":
        cmd_apply(args)
    elif args.subcommand == "undo":
        cmd_undo(args)
    elif args.subcommand == "purge":
        cmd_purge(args)
    elif args.subcommand == "history":
        cmd_history(args)
    elif args.subcommand == "categorize":
        cmd_categorize(args)
    elif args.subcommand == "search":
        cmd_search(args)
    elif args.subcommand == "list":
        cmd_list(args)
    elif args.subcommand == "show":
        cmd_show(args)
    elif args.subcommand == "serve":
        cmd_serve(args)
    elif args.subcommand == "drive-auth":
        cmd_drive_auth(args)
    elif args.subcommand == "scan-drive":
        cmd_scan_drive(args)
    elif args.subcommand == "confirm-drive":
        cmd_confirm_drive(args)


if __name__ == "__main__":
    main()

