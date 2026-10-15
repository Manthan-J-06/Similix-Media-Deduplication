# Argus — Local-First Image Deduplication System

> **Status:** See [Current Status](#current-status) section below.

---

## Goal

Detect exact and near-duplicate images across any format (HEIC, RAW, PNG, JPEG, WEBP, and others), group them into clusters, surface a **duplicate-confidence percentage** per cluster member, recommend exactly **one keeper** per cluster (the smallest file that is still visually equivalent to the best-quality copy), and let the user safely delete the rest.

---

## Principles

| Principle | Detail |
|---|---|
| **Local-first** | All processing and storage happens on-device; no data leaves the machine. |
| **Free & open-source** | Only free, open-source tools. No paid APIs or proprietary dependencies. |
| **Reversible by default** | Deletion moves files to trash and records every action in a ledger. Nothing is permanently erased. |
| **Cheap before expensive** | A staged funnel ensures full-resolution images are only loaded in the final stage, and only for candidate pairs that survive all earlier stages. |
| **Deterministic math** | All similarity scores and keeper decisions are produced by deterministic algorithms. The AI agent only orchestrates and explains — it never makes numerical judgements. |
| **Config-driven thresholds** | Every tuneable threshold lives in a single configuration file (`config.toml`). No magic numbers in code. |

---

## Constraints

- **Runtime:** Python 3.11+, SQLite, FastAPI.
- **Hardware target:** MacBook Air M-series, 16 GB RAM.
- **Memory discipline:** Never load full-resolution images unless Stage 4 explicitly requires it. Use thumbnails/proxies at ~256–512 px for all earlier stages.
- **Scale:** Must handle 5,000 to 500,000+ images without degradation.
  - No all-pairs comparison (O(n²) is prohibited).
  - Process in small, resumable batches.
  - Every image has a persistent status row in SQLite.
- **Source abstraction:** Image sources sit behind a single interface. Local folders are supported now; Google Drive (and other remotes) must be addable later without touching core logic.

---

## Pipeline — Staged Funnel (Cheap → Expensive)

Each stage receives only the candidates that survived the previous stage. Thresholds at every stage are defined in `config.toml`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 0 │ Size + MD5 exact match                                   │
│           │ Groups files by (file_size, md5). Identical files are   │
│           │ flagged as exact duplicates immediately.                 │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 1 │ Perceptual hash on thumbnail (~256 px)                   │
│           │ pHash / dHash / aHash. Hamming-distance threshold used  │
│           │ to build candidate pairs. Fast, in-process.             │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2 │ CLIP embedding + FAISS approximate nearest neighbours    │
│           │ Also produces zero-shot category labels per image.      │
│           │ Eliminates false positives from Stage 1.                │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 3 │ SSIM / quality check on candidate pairs only             │
│           │ Structural similarity + sharpness/noise scoring to      │
│           │ distinguish "same scene, different quality" from        │
│           │ "same scene, same quality".                             │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 4 │ Full-resolution check — keeper decision only             │
│           │ Loaded only for the surviving finalists. Used to pick   │
│           │ the definitive keeper (highest quality + smallest size  │
│           │ trade-off, per config weights).                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Clustering Rules

- **Verify against representative, not chain:** Each cluster member is compared to the cluster representative (the current best candidate), not to its nearest neighbour. This prevents transitive chaining where A≈B and B≈C incorrectly implies A≈C.
- **RAW + JPEG pairs:** A RAW file and its JPEG counterpart are tagged as *related files*, never as duplicates. They are excluded from deduplication and preserved together.

---

## Confidence Score

Each cluster member receives a **duplicate-confidence percentage** computed as a weighted combination of the signals from Stages 1–3:

```
confidence = w1·(phash_similarity) + w2·(clip_similarity) + w3·(ssim_score)
```

Weights (`w1`, `w2`, `w3`) are defined in `config.toml`. The keeper is always assigned 100%.

---

## Keeper Selection

The recommended keeper per cluster is the image that maximises:

```
keeper_score = quality_score − size_penalty
```

where `quality_score` is derived from Stage 3 (sharpness, noise, dynamic range) and `size_penalty` is a configurable factor applied to unnecessarily large files. Both factors are defined in `config.toml`.

RAW files are **never** selected as the keeper when a JPEG equivalent exists in the same cluster — they are moved to the *related files* group instead.

---

## Deletion Safety

1. **Trash-first:** Files are moved to the OS trash (macOS `~/.Trash`), never unlinked directly.
2. **Action ledger:** Every delete/move action is recorded in the SQLite `actions` table (timestamp, source path, destination path, operation, status).
3. **Undo:** The ledger enables a full restore of any batch delete operation.
4. **Dry-run mode:** A `--dry-run` flag previews all actions without executing them.

---

## Source Interface

All image sources implement the `ImageSource` abstract interface:

```python
class ImageSource(ABC):
    def list_files(self) -> Iterator[FileRecord]: ...
    def open_thumbnail(self, file_id: str, size: int) -> Image: ...
    def open_full(self, file_id: str) -> Image: ...
    def move_to_trash(self, file_id: str) -> ActionRecord: ...
```

| Source | Status |
|---|---|
| Local folder | ✅ Planned for v1 |
| Google Drive | 🔲 Planned for v2 |

---

## Repository Layout

```
argus/
├── core/           # Pipeline stages, scoring, clustering logic
├── sources/        # ImageSource implementations (local, gdrive, …)
├── db/             # SQLite schema, migrations, repository classes
├── agent/          # AI orchestration layer (explains, schedules, summarises)
├── api/            # FastAPI routes and request/response models
├── ui/             # Frontend (TBD: lightweight web UI)
├── bench/          # Benchmarks and performance regression tests
├── tests/          # Unit and integration tests
├── config.toml     # All tuneable thresholds and weights (single source of truth)
└── PROJECT.md      # This file
```

---

## Key Dependencies (Provisional)

| Library | Purpose |
|---|---|
| `Pillow` / `pillow-heif` | Image decoding incl. HEIC |
| `rawpy` | RAW format decoding |
| `imagehash` | Perceptual hashing (pHash, dHash) |
| `open-clip-torch` | CLIP embeddings (Apple Silicon MPS backend) |
| `faiss-cpu` | Approximate nearest-neighbour search |
| `scikit-image` | SSIM computation |
| `FastAPI` + `uvicorn` | REST API layer |
| `SQLite` (stdlib) | Persistent state, ledger, config cache |
| `pytest` | Test framework |

---

## Decisions Log

> _This section will be updated after every major design or implementation decision._

| # | Date | Decision | Rationale |
|---|---|---|---|
| 1 | 2026-10-14 | Generic `ImageSource` interface | Decouples core pipeline from local filesystem; enables Google Drive / S3 sources later. |
| 2 | 2026-10-14 | JPEG draft mode for thumbnail & pHash | Avoids decoding full 20MP+ pixel data in RAM; fulfills strict 16GB memory constraint. |
| 3 | 2026-10-14 | Rescanning skips unchanged files via `(modified_time, file_size)` | Massive speedup when re-running on 50,000+ files; only new or touched files are re-processed. |
| 4 | 2026-10-14 | Anti-chaining representative verification | Prevents false transitive duplicate clusters (A≈B and B≈C does not cluster C if C!=A). |
| 5 | 2026-10-14 | Ground-truth evaluation framework (`bench/evaluate.py`) | Measures precision and recall overall and per transform across thresholds; baseline establishes $T \in [8, 10]$ as optimal for Stage 1 (99.1% precision, 91.7% recall). |
| 6 | 2026-10-14 | Stage 2 CLIP Embeddings: `open_clip` `ViT-B-32` (`laion2b_s34b_b79k`) | **Model weights license:** Permissively licensed under **MIT License** (LAION community open weights). Accelerated via Apple Silicon **MPS** (72 img/s). Float16 L2-normalized 512-dim vectors stored as 1,024-byte BLOB. Rotation embedding cost benchmarked: 1.39x compute latency overhead, default kept off. |
| 7 | 2026-10-15 | Vector Index Architecture (`core/index.py`) | `VectorIndex` ABC with swappable implementations: `FlatIPIndex` (exact inner product for <200k images) and `HNSWIndex` (approximate graph index for >200k images) using `faiss.IndexIDMap2`. Incremental persistence ensures only newly embedded images are indexed. |
| 8 | 2026-10-15 | Anti-Chaining Clustering & Unified Cluster Table (`core/cluster.py`) | Anti-chaining representative verification: members must be directly similar to keeper (`cosine_threshold` / pHash), with mutual compatibility; failing members are split out and re-evaluated. Unified `clusters` table covers Stage 0 (MD5), Stage 1 (pHash), and Stage 2 (CLIP) with per-member `link_signal` and `confidence` score. OpenMP runtime collision guarded for macOS (`KMP_DUPLICATE_LIB_OK=TRUE`). |
| 9 | 2026-10-15 | Reproducible Benchmark Split & Held-Out Test Evaluation | Partitioned benchmark (200 originals, 4,000 images) into a 60% tuning set (120 originals, 2,400 images) and 40% held-out test set (80 originals, 1,600 images) split strictly by original image to prevent data leakage (`bench/split.json`, random seed `42`). Threshold tuning on the 60% set chose: pHash $T=8$, CLIP $\text{cos}=0.88$, and Combination $T=8, \text{cos}=0.90$. Held-out test set evaluation verified **99.60% precision**, **99.67% variant recall** (1,515/1,520 variants), and **99.41% pairwise recall** across all transformations. |

---

## Current Status

> _This section will be updated after every development step._

| Item | Status |
|---|---|
| Project spec | ✅ Written |
| Repo scaffold | ✅ Done — commit `8273ae7` |
| Stage 0 (MD5) | ✅ Done — commit `53b1b8b` |
| Stage 1 (pHash) | ✅ Done — commit `fb4ab5e` |
| Stage 2 (CLIP embeddings & FAISS clustering) | ✅ Done (`core/embeddings.py`, `core/index.py`, `core/cluster.py`, CLI `embed` & `cluster`) |
| Benchmark & Held-Out Test Evaluation | ✅ Done — 60/40 reproducible split (seed `42`), tuning on 60%, test on 40% (99.60% precision, 99.67% recall) |
| Stage 3 (SSIM) | 🔲 Next step |
| Stage 4 (full-res keeper) | 🔲 Not started |
| API layer | 🔲 Not started |
| UI | 🔲 Not started |
