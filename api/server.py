"""
api/server.py — FastAPI REST API for Argus.

Binds to 127.0.0.1 only. No CORS for external origins.
All file moves/deletions are strictly routed through core/ledger.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("argus.api")

# In-process synchronization locks for concurrent thumbnail generation
_thumb_locks: dict[tuple[int, int], threading.Lock] = {}
_thumb_global_lock = threading.Lock()


def _get_thumb_lock(image_id: int, size: int) -> threading.Lock:
    with _thumb_global_lock:
        key = (image_id, size)
        if key not in _thumb_locks:
            _thumb_locks[key] = threading.Lock()
        return _thumb_locks[key]


def _create_placeholder_jpeg() -> bytes:
    """Generate a small neutral gray JPEG placeholder for failed thumbnails."""
    buf = io.BytesIO()
    im = Image.new("RGB", (64, 64), color=(180, 180, 180))
    im.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


PLACEHOLDER_JPEG_BYTES = _create_placeholder_jpeg()

from core.categories import search_images_by_text
from core.config import load_config
from core.keeper import choose_keeper
from core.ledger import (
    ensure_ledger_tables,
    execute,
    get_history,
    plan,
    undo,
)
from db.connection import get_connection, init_db


class PlanRequest(BaseModel):
    mode: str = "conservative"
    approved_swap_cluster_ids: list[int] = Field(default_factory=list)
    excluded_ids: list[int] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class ApplyRequest(BaseModel):
    batch_id: str
    confirm: bool = False
    model_config = ConfigDict(extra="ignore")


class UndoRequest(BaseModel):
    batch_id: str
    model_config = ConfigDict(extra="ignore")


def get_db(request: Request):
    """Provide a per-request SQLite connection with row_factory enabled."""
    db_path = request.app.state.db_path
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def create_app(
    db_path: str | Path | None = None,
    trash_dir: str | Path | None = None,
    thumb_cache_dir: str | Path | None = None,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Create and configure the Argus FastAPI application."""
    if config is None:
        try:
            config = load_config(config_path) if config_path else load_config()
        except Exception:
            config = {}

    if db_path is None:
        db_path = os.environ.get("ARGUS_DB_PATH") or config.get("database", {}).get("path") or "argus.db"
    resolved_db_path = Path(db_path).expanduser().resolve()

    # Ensure schema and ledger tables are initialized
    init_conn = init_db(resolved_db_path)
    ensure_ledger_tables(init_conn)
    init_conn.close()

    if trash_dir is None:
        cfg_trash = config.get("ledger", {}).get("trash_dir")
        trash_dir = os.environ.get("ARGUS_TRASH_DIR") or cfg_trash or (Path.home() / ".argus" / "trash")
    resolved_trash_dir = Path(trash_dir).expanduser().resolve()

    if thumb_cache_dir is None:
        thumb_cache_dir = os.environ.get("ARGUS_THUMB_CACHE_DIR") or (Path.home() / ".argus" / "thumbnails")
    resolved_thumb_cache_dir = Path(thumb_cache_dir).expanduser().resolve()
    try:
        resolved_thumb_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    app = FastAPI(title="Argus API", version="0.1.0")

    # Strict local-only CORS: never allow external origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1",
            "http://127.0.0.1:8000",
            "http://localhost",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.db_path = resolved_db_path
    app.state.trash_dir = resolved_trash_dir
    app.state.thumb_cache_dir = resolved_thumb_cache_dir
    app.state.config = config

    ui_file = Path(__file__).resolve().parent.parent / "ui" / "index.html"

    @app.get("/")
    def serve_ui():
        """Serve the single-file Argus web UI."""
        if not ui_file.is_file():
            raise HTTPException(status_code=404, detail="UI index.html not found")
        return FileResponse(str(ui_file), media_type="text/html")

    @app.get("/clusters")
    def list_clusters(
        mode: str = Query("conservative"),
        category: Optional[str] = Query(None),
        min_savings: Optional[float] = Query(None),
        sort: str = Query("id"),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        if mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        c_rows = conn.execute(
            "SELECT id, representative_id, reference_uncertain, category FROM clusters ORDER BY id"
        ).fetchall()

        results: list[dict[str, Any]] = []
        for c_row in c_rows:
            cid = c_row["id"]
            ref_unc = c_row["reference_uncertain"] if "reference_uncertain" in c_row.keys() else 0
            members = conn.execute(
                """
                SELECT id, cluster_id, file_path, width, height, file_size, format, md5,
                       is_keeper, is_low_confidence, ssim, ssim_color, sharpness_ratio,
                       pixel_ratio, bytes_ratio, quality_status, review_flag, delete_safe,
                       reason, source_likelihood, jpeg_quality_est, category_top1,
                       category_top1_prob, match_pct, resolution_pct, confidence
                FROM images
                WHERE cluster_id = ?
                ORDER BY id
                """,
                (cid,),
            ).fetchall()

            if len(members) < 2:
                continue

            member_dicts = [dict(m) for m in members]
            keeper_res = choose_keeper(
                member_dicts,
                mode=mode,
                config=request.app.state.config,
                reference_uncertain=ref_unc,
            )

            keeper_id = keeper_res["keeper_id"]
            del_safe_ids = set(keeper_res["delete_safe_ids"])
            review_ids = set(keeper_res["review_ids"])
            swap_prop = keeper_res.get("swap_proposal")

            del_safe_members = [m for m in member_dicts if m["id"] in del_safe_ids]
            savings_bytes = sum(m["file_size"] or 0 for m in del_safe_members)
            total_bytes = sum(m["file_size"] or 0 for m in member_dicts)
            savings_ratio = round(savings_bytes / total_bytes, 4) if total_bytes > 0 else 0.0

            keeper_m = next((m for m in member_dicts if m["id"] == keeper_id), None)
            cluster_cat = c_row["category"] or (keeper_m.get("category_top1") if keeper_m else None)

            # Category filter
            if category is not None and (cluster_cat or "").lower() != category.lower():
                continue

            # Minimum savings filter
            if min_savings is not None:
                if 0 < min_savings <= 1.0:
                    if savings_ratio < min_savings:
                        continue
                else:
                    if savings_bytes < min_savings:
                        continue

            results.append({
                "id": cid,
                "keeper": {
                    "id": keeper_id,
                    "file_path": keeper_m["file_path"] if keeper_m else "",
                    "file_size": keeper_m["file_size"] if keeper_m else 0,
                    "format": keeper_m["format"] if keeper_m else "",
                },
                "savings_bytes": savings_bytes,
                "reclaimable_bytes": savings_bytes,
                "savings_ratio": savings_ratio,
                "member_count": len(member_dicts),
                "delete_safe_count": len(del_safe_ids),
                "review_count": len(review_ids),
                "category": cluster_cat,
                "has_swap_proposal": swap_prop is not None,
                "swap_proposal": swap_prop,
            })

        # Sorting
        if sort in ("savings", "-savings"):
            results.sort(key=lambda x: x["savings_bytes"], reverse=True)
        elif sort == "+savings":
            results.sort(key=lambda x: x["savings_bytes"], reverse=False)
        elif sort in ("count", "counts", "-count", "-counts"):
            results.sort(key=lambda x: x["member_count"], reverse=True)
        elif sort in ("+count", "+counts"):
            results.sort(key=lambda x: x["member_count"], reverse=False)
        elif sort in ("category", "+category"):
            results.sort(key=lambda x: (x["category"] or "").lower())
        elif sort == "-category":
            results.sort(key=lambda x: (x["category"] or "").lower(), reverse=True)
        elif sort == "-id":
            results.sort(key=lambda x: x["id"], reverse=True)
        else:
            results.sort(key=lambda x: x["id"])

        return results

    @app.get("/clusters/{cluster_id}")
    def get_cluster(
        cluster_id: int,
        mode: str = Query("conservative"),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        c_row = conn.execute(
            "SELECT id, representative_id, reference_uncertain, category FROM clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        if not c_row:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        members = conn.execute(
            """
            SELECT id, cluster_id, file_path, width, height, file_size, format, md5,
                   is_keeper, is_low_confidence, ssim, ssim_color, sharpness_ratio,
                   pixel_ratio, bytes_ratio, quality_status, review_flag, delete_safe,
                   reason, source_likelihood, jpeg_quality_est, category_top1,
                   category_top1_prob, match_pct, resolution_pct, confidence
            FROM images
            WHERE cluster_id = ?
            ORDER BY id
            """,
            (cluster_id,),
        ).fetchall()

        if not members:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} has no members")

        member_dicts = [dict(m) for m in members]
        ref_unc = c_row["reference_uncertain"] if "reference_uncertain" in c_row.keys() else 0
        keeper_res = choose_keeper(
            member_dicts,
            mode=mode,
            config=request.app.state.config,
            reference_uncertain=ref_unc,
        )

        keeper_id = keeper_res["keeper_id"]
        del_safe_ids = set(keeper_res["delete_safe_ids"])
        review_ids = set(keeper_res["review_ids"])
        reasons = keeper_res["reasons"]

        formatted_members = []
        for m in member_dicts:
            mid = m["id"]
            formatted_members.append({
                "id": mid,
                "file_path": m["file_path"],
                "size": m["file_size"],
                "width": m["width"],
                "height": m["height"],
                "format": m["format"],
                "match_pct": m["match_pct"],
                "resolution_pct": m["resolution_pct"],
                "confidence": m["confidence"],
                "category": m["category_top1"],
                "category_prob": m["category_top1_prob"],
                "is_recommended_keeper": mid == keeper_id,
                "delete_safe": mid in del_safe_ids,
                "review": mid in review_ids,
                "reason": reasons.get(mid, ""),
            })

        keeper_m = next((m for m in formatted_members if m["is_recommended_keeper"]), None)
        cluster_cat = c_row["category"] or (keeper_m.get("category") if keeper_m else None)

        return {
            "id": c_row["id"],
            "mode": mode,
            "category": cluster_cat,
            "reference_uncertain": bool(ref_unc),
            "swap_proposal": keeper_res.get("swap_proposal"),
            "members": formatted_members,
        }

    @app.get("/thumb/{image_id}")
    def get_thumbnail(
        image_id: int,
        size: int = Query(256, ge=32, le=2048),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ):
        """Serve a cached JPEG thumbnail for the given image ID. Never accepts file paths."""
        row = conn.execute("SELECT id, file_path FROM images WHERE id = ?", (image_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

        orig_path = Path(row["file_path"])
        if not orig_path.exists():
            logger.warning(f"Image {image_id} source file not found on disk: {orig_path}")
            return Response(content=PLACEHOLDER_JPEG_BYTES, media_type="image/jpeg", status_code=200)

        cache_dir = request.app.state.thumb_cache_dir
        cache_path = cache_dir / f"{image_id}_{size}.jpg"

        # Check existing valid cache
        if cache_path.is_file():
            try:
                st = cache_path.stat()
                if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                    return FileResponse(str(cache_path), media_type="image/jpeg")
            except OSError:
                pass

        # Concurrency synchronization per (image_id, size)
        lock = _get_thumb_lock(image_id, size)
        with lock:
            # Re-check inside lock in case another thread just completed generation
            if cache_path.is_file():
                try:
                    st = cache_path.stat()
                    if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                        return FileResponse(str(cache_path), media_type="image/jpeg")
                except OSError:
                    pass

            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_dir / f".tmp_{image_id}_{size}_{uuid.uuid4().hex}.jpg"
            try:
                with Image.open(orig_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.thumbnail((size, size), Image.Resampling.LANCZOS)
                    img.save(str(tmp_path), format="JPEG", quality=85)
                os.replace(str(tmp_path), str(cache_path))
            except Exception as exc:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                logger.error(f"Thumbnail generation failed for image {image_id} ({orig_path}): {exc}")
                return Response(content=PLACEHOLDER_JPEG_BYTES, media_type="image/jpeg", status_code=200)

            return FileResponse(str(cache_path), media_type="image/jpeg")

    @app.get("/search")
    def search_images(
        q: str = Query(..., description="Natural language search query"),
        limit: int = Query(10, ge=1, le=100),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        if not q.strip():
            return []
        try:
            return search_images_by_text(
                query=q,
                conn=conn,
                config=request.app.state.config,
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    @app.get("/categories")
    def get_categories(
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, int]:
        rows = conn.execute(
            """
            SELECT category_top1, COUNT(*) AS count
            FROM images
            WHERE category_top1 IS NOT NULL
            GROUP BY category_top1
            ORDER BY count DESC
            """
        ).fetchall()
        return {r["category_top1"]: r["count"] for r in rows}

    @app.post("/plan")
    def create_plan(
        req: PlanRequest,
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if req.mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        # Enforce that all excluded_ids are valid members of a cluster
        if req.excluded_ids:
            placeholders = ",".join("?" for _ in req.excluded_ids)
            rows = conn.execute(
                f"SELECT id FROM images WHERE id IN ({placeholders}) AND cluster_id IS NOT NULL",
                req.excluded_ids,
            ).fetchall()
            found_ids = {r[0] for r in rows}
            invalid_ids = [mid for mid in req.excluded_ids if mid not in found_ids]
            if invalid_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image ID(s) not currently a member of a cluster: {invalid_ids}",
                )

        if req.approved_swap_cluster_ids:
            c_placeholders = ",".join("?" for _ in req.approved_swap_cluster_ids)
            c_rows = conn.execute(
                f"SELECT id FROM clusters WHERE id IN ({c_placeholders})",
                req.approved_swap_cluster_ids,
            ).fetchall()
            found_c_ids = {r[0] for r in c_rows}
            invalid_c = [cid for cid in req.approved_swap_cluster_ids if cid not in found_c_ids]
            if invalid_c:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cluster ID(s) not found: {invalid_c}",
                )

        try:
            summary = plan(
                conn=conn,
                mode=req.mode,
                approved_swap_cluster_ids=req.approved_swap_cluster_ids,
                excluded_ids=req.excluded_ids,
                config=request.app.state.config,
                trash_dir=request.app.state.trash_dir,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "batch_id": summary.batch_id,
            "mode": summary.mode,
            "file_count": summary.file_count,
            "bytes_freed": summary.bytes_freed,
            "trash_dir": summary.trash_dir,
            "per_cluster_keeper": summary.per_cluster_keeper,
            "entries": [
                {
                    "image_id": e.image_id,
                    "cluster_id": e.cluster_id,
                    "keeper_id": e.keeper_id,
                    "original_path": e.original_path,
                    "file_size": e.file_size,
                    "reason": e.reason,
                    "is_manual": e.is_manual,
                }
                for e in summary.entries
            ],
        }

    @app.post("/apply")
    def apply_batch(
        req: ApplyRequest,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if not req.confirm:
            raise HTTPException(
                status_code=400,
                detail="Execution requires explicit confirmation (confirm: true)",
            )

        batch = conn.execute(
            "SELECT status FROM ledger_batches WHERE batch_id = ?",
            (req.batch_id,),
        ).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch '{req.batch_id}' not found in ledger")
        if batch["status"] != "planned":
            raise HTTPException(
                status_code=400,
                detail=f"Batch '{req.batch_id}' has status '{batch['status']}'; only 'planned' batches can be executed",
            )

        try:
            res = execute(conn, batch_id=req.batch_id, confirm=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "batch_id": res.batch_id,
            "total_planned": res.total_planned,
            "moved_count": res.moved_count,
            "refused_count": res.refused_count,
            "failed_count": res.failed_count,
            "bytes_moved": res.bytes_moved,
            "refusals": res.refusals,
            "failures": res.failures,
        }

    @app.post("/undo")
    def undo_batch(
        req: UndoRequest,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        batch = conn.execute(
            "SELECT status FROM ledger_batches WHERE batch_id = ?",
            (req.batch_id,),
        ).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch '{req.batch_id}' not found in ledger")

        try:
            res = undo(conn, batch_id=req.batch_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "batch_id": res.batch_id,
            "restored_count": res.restored_count,
            "conflict_count": res.conflict_count,
            "conflicts": res.conflicts,
            "restored_files": res.restored_files,
        }

    @app.get("/history")
    def batch_history(
        limit: int = Query(50, ge=1, le=500),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        return get_history(conn, limit=limit)

    return app


# Module-level app instance with protected initialization
try:
    app = create_app()
except Exception:
    app = FastAPI(title="Argus API", version="0.1.0")

