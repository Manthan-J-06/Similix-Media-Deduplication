"""
core/categories.py — Stage 5: Zero-shot semantic categorization and text search.

Features:
- Encodes label prompt templates using open_clip text encoder (same model as image embeddings).
- Averages prompt embeddings per label, L2-normalizes, and caches text embeddings.
- Classifies using upright clip_vector only (never rotated vectors).
- Computes softmax probabilities with configurable temperature.
- Flags near-ties or low-confidence predictions as 'uncertain' using uncertain_margin and min_top_prob.
- Stores category_top1, category_top1_prob, and category_top3 (JSON array of label/prob pairs).
- Assigns cluster-level category based on keeper's label, falling back to member majority vote.
- Fast vector text search against stored upright clip_vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
from typing import Any, Callable

import numpy as np
import open_clip
import torch

from core.embeddings import CLIPEmbedder, deserialize_embedding, get_device


DEFAULT_SCENE_LABELS = [
    "indoor",
    "landscape",
    "architecture",
    "workplace",
    "studio",
]

DEFAULT_SUBJECT_LABELS = [
    "portrait",
    "animal",
    "food",
    "vehicle",
    "document",
    "screenshot",
    "technology",
    "artifact",
    "people_working",
    "other",
]


@dataclass
class CategoryConfig:
    temperature: float = 0.05
    uncertain_margin: float = 0.05
    min_top_prob: float = 0.20
    labels: list[str] = field(default_factory=list)
    templates: dict[str, list[str]] = field(default_factory=dict)
    scene_labels: list[str] = field(default_factory=list)
    subject_labels: list[str] = field(default_factory=list)


def parse_category_config(config: dict) -> CategoryConfig:
    """Parse [categories] section from config.toml with support for two-level scene & subject labels."""
    cat_cfg = config.get("categories", {})
    temperature = float(cat_cfg.get("temperature", 0.05))
    uncertain_margin = float(cat_cfg.get("uncertain_margin", 0.05))
    min_top_prob = float(cat_cfg.get("min_top_prob", 0.20))

    scene_cfg = cat_cfg.get("scene_labels")
    scene_labels = [str(s) for s in scene_cfg] if isinstance(scene_cfg, list) else list(DEFAULT_SCENE_LABELS)

    subject_cfg = cat_cfg.get("subject_labels")
    subject_labels = [str(s) for s in subject_cfg] if isinstance(subject_cfg, list) else list(DEFAULT_SUBJECT_LABELS)

    templates_cfg = cat_cfg.get("templates")
    labels_cfg = cat_cfg.get("labels")

    templates: dict[str, list[str]] = {}
    if isinstance(templates_cfg, dict):
        for label, tmpls in templates_cfg.items():
            if isinstance(tmpls, list):
                templates[str(label)] = [str(t) for t in tmpls]
            elif isinstance(tmpls, dict) and "templates" in tmpls:
                templates[str(label)] = [str(t) for t in tmpls["templates"]]
    elif isinstance(labels_cfg, dict):
        for label, tmpls in labels_cfg.items():
            if isinstance(tmpls, list):
                templates[str(label)] = [str(t) for t in tmpls]
            elif isinstance(tmpls, dict) and "templates" in tmpls:
                templates[str(label)] = [str(t) for t in tmpls["templates"]]

    if isinstance(labels_cfg, list):
        labels = [str(l) for l in labels_cfg]
    else:
        # Merge scene labels, subject labels, and any custom templates preserving order
        seen = set()
        labels = []
        for l in scene_labels + subject_labels + list(templates.keys()):
            if l not in seen:
                seen.add(l)
                labels.append(l)

    return CategoryConfig(
        temperature=temperature,
        uncertain_margin=uncertain_margin,
        min_top_prob=min_top_prob,
        labels=labels,
        templates=templates,
        scene_labels=scene_labels,
        subject_labels=subject_labels,
    )


class TextEmbeddingCache:
    """Encodes and caches L2-normalized text prompt embeddings for label zero-shot classification."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: torch.device | None = None,
        embedder: CLIPEmbedder | None = None,
    ):
        if embedder is not None:
            self.model = embedder.model
            self.device = embedder.device
            self.model_name = embedder.model_name
            self.pretrained = embedder.pretrained
        else:
            self.device = device or get_device()
            self.model_name = model_name
            self.pretrained = pretrained
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
            )
            self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self._cache: dict[str, np.ndarray] = {}

    def encode_label(self, label: str, prompt_templates: list[str]) -> np.ndarray:
        """
        Encode prompt templates for a label, average them, and return a unit L2-normalized vector.

        Returns:
            np.ndarray of shape (512,), dtype float32, unit L2 norm.
        """
        if not prompt_templates:
            prompt_templates = [f"a photo of a {label}"]

        cache_key = f"{label}::" + "|".join(prompt_templates)
        if cache_key in self._cache:
            return self._cache[cache_key]

        tokens = self.tokenizer(prompt_templates).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_feature = text_features.mean(dim=0, keepdim=True)
            mean_feature = mean_feature / mean_feature.norm(dim=-1, keepdim=True)
            vec = mean_feature.squeeze(0).cpu().to(torch.float32).numpy()

        # Free tensors
        del tokens, text_features, mean_feature
        if self.device.type == "mps":
            torch.mps.empty_cache()

        self._cache[cache_key] = vec
        return vec

    def build_label_matrix(self, labels: list[str], templates: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
        """
        Build text embedding matrix for all labels.

        Returns:
            (matrix of shape (num_labels, 512), list of label names)
        """
        valid_labels = []
        vectors = []
        for l in labels:
            tmpls = templates.get(l, [f"a photo of a {l}"])
            vec = self.encode_label(l, tmpls)
            vectors.append(vec)
            valid_labels.append(l)

        if not vectors:
            return np.empty((0, 512), dtype=np.float32), []

        matrix = np.vstack(vectors).astype(np.float32)
        return matrix, valid_labels

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode an arbitrary text query for semantic search.

        Returns:
            np.ndarray of shape (512,), dtype float32, unit L2 norm.
        """
        tokens = self.tokenizer([query]).to(self.device)
        with torch.inference_mode():
            q_feat = self.model.encode_text(tokens)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
            vec = q_feat.squeeze(0).cpu().to(torch.float32).numpy()

        del tokens, q_feat
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return vec


def classify_vector(
    clip_vec: np.ndarray,
    text_matrix: np.ndarray,
    label_names: list[str],
    temperature: float = 0.05,
    uncertain_margin: float = 0.05,
    min_top_prob: float = 0.20,
) -> tuple[str, float, list[dict[str, Any]]]:
    """
    Classify a single upright image CLIP vector against label text embeddings.

    Returns:
        (category_top1, category_top1_prob, category_top3)
        where category_top1 is either a label name or 'uncertain'.
    """
    if len(label_names) == 0 or text_matrix.shape[0] == 0:
        return "uncertain", 0.0, []

    # Ensure float32 and unit norm
    vec = clip_vec.astype(np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    # Cosine similarities
    sims = np.dot(text_matrix, vec)  # shape (num_labels,)

    # Softmax with temperature
    temp = max(float(temperature), 1e-4)
    logits = sims / temp
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / (np.sum(exp_logits) or 1.0)

    ranked_idx = np.argsort(probs)[::-1]
    top1_idx = ranked_idx[0]
    top1_label = label_names[top1_idx]
    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[ranked_idx[1]]) if len(ranked_idx) > 1 else 0.0

    # Uncertain rule:
    # 1. Margin between top 1 and top 2 is smaller than uncertain_margin
    # 2. Absolute probability of top 1 is below min_top_prob
    is_uncertain = ((top1_prob - top2_prob) < uncertain_margin) or (top1_prob < min_top_prob)
    final_label = "uncertain" if is_uncertain else top1_label

    top3 = [
        {"label": label_names[idx], "prob": round(float(probs[idx]), 4)}
        for idx in ranked_idx[: min(3, len(ranked_idx))]
    ]

    return final_label, round(top1_prob, 4), top3


@dataclass
class CategorizeResult:
    images_categorized: int = 0
    images_skipped: int = 0
    clusters_updated: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    uncertain_count: int = 0


def update_cluster_categories(conn: sqlite3.Connection) -> int:
    """
    Update cluster-level category across all clusters:
    1. If manual_category is set on the cluster, preserve it as the authoritative label.
    2. Otherwise, use the keeper's category_top1 if present.
    3. Fall back to majority vote of cluster members' category_top1.
    """
    cluster_cols = {row[1] for row in conn.execute("PRAGMA table_info(clusters)").fetchall()}
    has_manual = "manual_category" in cluster_cols
    manual_clause = "manual_category," if has_manual else "NULL AS manual_category,"
    clusters = conn.execute(f"SELECT id, representative_id, {manual_clause} category FROM clusters ORDER BY id").fetchall()
    updated = 0

    cols = {row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()}
    has_rec = "is_recommended_keeper" in cols
    rec_col = "is_recommended_keeper" if has_rec else "0 AS is_recommended_keeper"

    for c in clusters:
        cid = c["id"]
        rep_id = c["representative_id"]
        manual_cat = c["manual_category"] if has_manual and c["manual_category"] else None

        if manual_cat:
            conn.execute("UPDATE clusters SET category = ? WHERE id = ?", (manual_cat, cid))
            updated += 1
            continue

        members = conn.execute(
            f"""
            SELECT id, is_keeper, {rec_col}, category_top1
            FROM images
            WHERE cluster_id = ?
            ORDER BY id
            """,
            (cid,),
        ).fetchall()

        if not members:
            continue

        cluster_category: str | None = None

        # 1. Try to find the keeper member
        keeper = None
        for m in members:
            if rep_id is not None and m["id"] == rep_id:
                keeper = m
                break
            if m["is_keeper"] == 1 or m["is_recommended_keeper"] == 1:
                keeper = m
                break

        if keeper is not None and keeper["category_top1"]:
            cluster_category = keeper["category_top1"]
        else:
            # 2. Fall back to majority vote among members with a non-null category
            valid_cats = [m["category_top1"] for m in members if m["category_top1"]]
            if valid_cats:
                # Count frequencies
                from collections import Counter
                counts = Counter(valid_cats)
                cluster_category = counts.most_common(1)[0][0]

        if cluster_category is not None:
            conn.execute(
                "UPDATE clusters SET category = ? WHERE id = ?",
                (cluster_category, cid),
            )
            updated += 1

    conn.commit()
    return updated


def run_categorization(
    conn: sqlite3.Connection,
    config: dict,
    force: bool = False,
    embedder: CLIPEmbedder | None = None,
    text_cache: TextEmbeddingCache | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CategorizeResult:
    """
    Categorize all images with upright clip_vector into zero-shot semantic categories.

    Resumable by default: skips images where category_top1 IS NOT NULL unless force=True.
    Classifies using UPRIGHT clip_vector ONLY (never clip_vector_90/180/270).
    """
    cat_cfg = parse_category_config(config)
    if not cat_cfg.labels:
        raise ValueError(
            "No categories configured in config.toml [categories]. "
            "Please provide [categories.templates] with label templates."
        )

    # Setup text embeddings cache
    stage2_cfg = config.get("stage2", {})
    model_name = stage2_cfg.get("clip_model", "ViT-B-32")
    pretrained = stage2_cfg.get("clip_pretrained", "laion2b_s34b_b79k")

    if text_cache is None:
        text_cache = TextEmbeddingCache(
            model_name=model_name,
            pretrained=pretrained,
            embedder=embedder,
        )

    text_matrix, label_names = text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)

    # Query images with upright clip_vector
    if force:
        where_clause = "clip_vector IS NOT NULL"
    else:
        where_clause = "clip_vector IS NOT NULL AND category_top1 IS NULL"

    pending_rows = conn.execute(
        f"SELECT id, clip_vector FROM images WHERE {where_clause} ORDER BY id"
    ).fetchall()

    already_done_count = conn.execute(
        "SELECT COUNT(*) FROM images WHERE category_top1 IS NOT NULL"
    ).fetchone()[0]

    total = len(pending_rows)
    counts_by_cat: dict[str, int] = {}
    uncertain_count = 0

    batch_size = 200
    for idx in range(0, total, batch_size):
        chunk = pending_rows[idx : idx + batch_size]
        updates = []
        for row in chunk:
            img_id = row["id"]
            blob = row["clip_vector"]
            # Strictly decode upright clip_vector
            vec = deserialize_embedding(blob)
            cat_top1, cat_top1_prob, cat_top3 = classify_vector(
                vec,
                text_matrix,
                label_names,
                temperature=cat_cfg.temperature,
                uncertain_margin=cat_cfg.uncertain_margin,
                min_top_prob=cat_cfg.min_top_prob,
            )
            cat_top2 = cat_top3[1]["label"] if len(cat_top3) > 1 else None
            cat_top2_prob = cat_top3[1]["prob"] if len(cat_top3) > 1 else None
            updates.append((cat_top1, cat_top1_prob, cat_top2, cat_top2_prob, json.dumps(cat_top3), img_id))

            counts_by_cat[cat_top1] = counts_by_cat.get(cat_top1, 0) + 1
            if cat_top1 in ("uncertain", "Needs label"):
                uncertain_count += 1

        conn.executemany(
            """
            UPDATE images
            SET category_top1 = ?,
                category_top1_prob = ?,
                category_top2 = ?,
                category_top2_prob = ?,
                category_top3 = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            updates,
        )
        conn.commit()

        if progress_callback:
            progress_callback(min(idx + len(chunk), total), total)

    # Update cluster-level categories
    clusters_updated = update_cluster_categories(conn)

    skipped = 0 if force else already_done_count
    return CategorizeResult(
        images_categorized=total,
        images_skipped=skipped,
        clusters_updated=clusters_updated,
        category_counts=counts_by_cat,
        uncertain_count=uncertain_count,
    )


def search_images_by_text(
    query: str,
    conn: sqlite3.Connection,
    config: dict,
    limit: int = 10,
    text_cache: TextEmbeddingCache | None = None,
    embedder: CLIPEmbedder | None = None,
) -> list[dict[str, Any]]:
    """
    Search images by natural language text query using CLIP cosine similarity against upright clip_vector.

    Returns:
        List of dicts containing image metadata and similarity score, sorted descending.
    """
    stage2_cfg = config.get("stage2", {})
    model_name = stage2_cfg.get("clip_model", "ViT-B-32")
    pretrained = stage2_cfg.get("clip_pretrained", "laion2b_s34b_b79k")

    if text_cache is None:
        text_cache = TextEmbeddingCache(
            model_name=model_name,
            pretrained=pretrained,
            embedder=embedder,
        )

    q_vec = text_cache.encode_query(query).astype(np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    rows = conn.execute(
        """
        SELECT id, file_path, file_size, width, height, format,
               category_top1, category_top1_prob, cluster_id, clip_vector
        FROM images
        WHERE clip_vector IS NOT NULL
        """
    ).fetchall()

    results = []
    for r in rows:
        img_vec = deserialize_embedding(r["clip_vector"]).astype(np.float32)
        norm = np.linalg.norm(img_vec)
        if norm > 0:
            img_vec = img_vec / norm
        score = float(np.dot(q_vec, img_vec))

        results.append({
            "id": r["id"],
            "file_path": r["file_path"],
            "file_size": r["file_size"],
            "width": r["width"],
            "height": r["height"],
            "format": r["format"],
            "category_top1": r["category_top1"],
            "category_top1_prob": r["category_top1_prob"],
            "cluster_id": r["cluster_id"],
            "score": round(score, 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
