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
| 10 | 2026-10-15 | Rotation Embeddings Default-On & Best-Matching Cluster Assignment | Set `enable_rotation_embeddings = true` by default in `config.toml` (accepting the 2.7x embedding time cost: 37.3s vs 13.8s per 1,000 images) to achieve 100% recall on 90°/180°/270° rotated duplicates. Fixed combination mode Stage 2 cluster assignment by ranking candidate matches to existing representatives by direct similarity descending rather than arbitrary database iteration order, eliminating greedy misassignment of rotated variants to visually adjacent clusters and raising rotated variant recall from 98.8% to 100.0%. |
| 11 | 2026-10-15 | Stage 3 Geometric Verification via SIFT Keypoint Matching & RANSAC | **Licensing:** SIFT US Patent 6,711,293 expired in March 2020; OpenCV moved SIFT to the main repository under the **Apache 2.0 License** in OpenCV 4.4.0 (July 2020), making it fully patent-free and permissively licensed worldwide. Implemented `core/verify.py` using SIFT keypoint extraction with RANSAC affine/homography estimation on ~512px proxies. Cluster members are verified against the cluster representative (`min_inliers = 20`, `min_inlier_ratio = 0.65`); failing members are split out and singletons disbanded. Tolerates crops, rescaling, and 90°/180°/270° rotations while cleanly rejecting cross-scene false positive pairs (e.g., `picsum_0008` vs `0009`, `picsum_0001` vs `0005`). Achieves **99.86% precision** on the held-out test set with minimal computational overhead (**17.26s per 1,000 images**). |
| 12 | 2026-10-15 | Stage 3 Robustness: Re-evaluation of Split Members, Inconclusive Verdict, and Keypoint Caching | Resolved Stage 3 held-out recall reduction (from 100% to 96.32%) without changing verification thresholds: (1) Diagnosed 56 missed variants into three root causes: disbanded low-texture scenes (`picsum_0063`, rep keypoints=13), downsampled tiny thumbnails (`resize_128` with <100 keypoints), and unclustered split scene members (`picsum_0009` split from `picsum_0008`); (2) Implemented re-evaluation of split members against surviving cluster representatives and against each other, regrouping separated scenes into distinct new clusters (`picsum_0009` successfully forms its own cluster); (3) Added an `'inconclusive'` verdict for image pairs with $< N$ keypoints (`min_keypoints = 100` in `config.toml`), which preserves their existing cluster, flags them with `is_low_confidence = 1` in SQLite, and excludes them from auto-deletion; (4) Added binary keypoint/descriptor caching in SQLite table `keypoint_cache` so representatives and members are never re-detected. Held-out test set performance restored to **99.60% precision**, **99.74% variant recall** (1,516/1,520 variants), with 171 low-confidence members, at **18.29s per 1,000 images**. |
| 13 | 2026-10-15 | Combined-Mode Threshold Selection Rule on Tuning Set | Updated `bench/evaluate.py` to select combined-mode thresholds strictly on the 60% tuning set (random seed `42`) by the rule: **maximize variant recall subject to $\text{precision} \ge 99.5\%$ with cosine threshold $\ge 0.90$**. Search over Hamming thresholds $T \in [4, 14]$ and cosine thresholds $\text{cos} \in [0.90, 0.96]$ chose **pHash $T = 8$** and **CLIP $\text{cos} = 0.90$** (tuning set performance: **100.00% precision**, **100.00% variant recall**). Single evaluation on the 40% held-out test set achieved **100.00% precision** and **99.87% variant recall** pre-Stage 3, and **99.87% precision**, **99.74% variant recall** post-Stage 3 with 172 low-confidence members, verifying strong generalization without threshold overfitting. |
| 14 | 2026-10-16 | Threshold Synchronization between Benchmark and Active Config | Synchronized `config.toml` to the tuned parameters from Decision 13 (`hamming_threshold = 8`, `cosine_threshold = 0.90`). Updated `bench/evaluate.py` to serialize tuned parameters to a single source of truth (`bench/tuned.json`) and emit a prominent terminal warning if active `config.toml` thresholds diverge from tuned values. Added unit tests in `tests/test_config.py` enforcing that `config.toml` matches `bench/tuned.json`. Reran Stage 0–3 CLI on `~/dedupe_test/photos`; Stage 2 formed a tentative candidate pair `(picsum_0001, picsum_0005)` under `cos = 0.90`, but Stage 3 SIFT/RANSAC geometric verification rejected the pair and disbanded the cluster, keeping the final output clusters completely unchanged (Cluster 1: 3 images, Cluster 2: 6 images). |
| 15 | 2026-10-16 | Stage 4 Quality Scoring & Proxy Geometric Alignment (`core/quality.py`) | Implemented proxy-based quality evaluation and geometric alignment: (1) Cluster reference member selected by primary: highest pixel count (`width * height`), secondary: least compressed (`file_size / (width * height)` bytes per pixel), tertiary: shortest file path; (2) Members aligned to reference using Stage 3 cached geometric transforms in `pair_transforms` (recomputed via SIFT + RANSAC if missing, supporting rotation and rescale); (3) Unalignable members marked `quality_status = 'unalignable'` and strictly excluded from auto-delete recommendations; (4) Metrics computed on ~1024px proxies: Grayscale SSIM (`scikit-image`), color SSIM (`channel_axis=2`), Laplacian variance sharpness ratio, pixel ratio, and bytes ratio; (5) All thresholds configured in `config.toml` under `[quality]`; (6) Synthetic test suite added in `tests/test_quality.py` for identical image (SSIM ~ 1.0), JPEG q30 (lower SSIM), blurred copy (low sharpness ratio), 90° rotated copy (aligned SSIM stays high), and unrelated image (rejected as unalignable). Evaluated on `~/dedupe_test/photos`: Cluster 1 chose `IMG_B.jpg`, Cluster 2 chose `A_conv.png` (highest bpp: 1.355 vs 0.147 bpp). All 8 non-reference members aligned successfully with zero unalignable failures. Keeper selection intentionally deferred. |
| 16 | 2026-10-16 | Stage 4 Keeper Selection & Fused Duplicate Confidence Calibration (`core/keeper.py`) | Implemented pure-function keeper selection and deletion recommendation rule: (1) Instant UI recomputation via `choose_keeper(cluster_members, mode)` supporting `conservative` (SSIM 0.98/0.97, pixel floor 0.90, blur=false), `balanced` (SSIM 0.95/0.94, pixel floor 0.60, blur=false), and `aggressive` (SSIM 0.92/0.90, pixel floor 0.25, blur=true); (2) Keeper chosen as smallest file size among qualifying candidates, breaking ties deterministically by higher pixel count, then shorter path length; if no other candidate qualifies, the reference is kept; (3) Clean partitioning into `delete_safe_ids` (byte-identical copies or verified aligned members) and `review_ids` (unalignable, low-confidence, or inconclusive members); guaranteed safety invariant that keeper is never delete-safe; (4) SQLite images table extended to store `is_recommended_keeper`, `delete_safe`, `review_flag`, `quality_retained_pct`, `bytes_saved_if_deleted`, and `reason`; (5) Fused duplicate confidence model fitted via regularized logistic regression strictly on the 60% tuning set using features: pHash Hamming distance ($\beta = -0.2796$), best CLIP cosine across 0°/90°/180°/270° ($\beta = +13.4554$), SIFT inlier ratio ($\beta = +5.6825$), and log inlier count ($\beta = +0.8481$), with exact MD5 matches fixed at 100%. Calibrated on the 40% held-out test set (16,558 pairs) yielding an exceptional Brier score of **0.00443** (0.03% calibration error in top [0.9, 1.0] bucket). Model persisted to `models/duplicate_confidence.json`. |
| 17 | 2026-10-16 | Keeper Safety Guards, Minimum Savings Ratio & Conservative Non-Ref Rule | Refined `core/keeper.py` with multi-layered safety guards: (1) **Keeper Quality & Resolution Guard**: A member is `delete_safe` only if the keeper is at least as good (`keeper_pixel_count >= member_pixel_count` AND `keeper_quality >= member_quality - 0.005`). Any member superior in either resolution or visual fidelity goes to `review_ids` with reason `"better than recommended keeper"`, preventing lower-res/lossy keepers from automatically purging high-res originals; (2) **Minimum Savings Ratio**: Configured `min_saving_ratio` in `config.toml` (`conservative: 0.0`, `balanced: 0.10`, `aggressive: 0.30`); non-reference candidates become keeper only if `file_size <= reference_size * (1 - min_saving_ratio)`; (3) **Conservative Mode Non-Ref Safety Rule**: In conservative mode, candidates require `pixel_ratio >= 0.90`. When the keeper is not the reference copy, only byte-identical members are eligible for automatic `delete_safe`; all non-identical variants are held for user review; (4) Unit tests added to `tests/test_keeper.py` for aggressive mode min saving, superior copy protection, crafted 4-member cluster mode divergence, and conservative non-ref review; (5) Validated on `~/dedupe_test/demo` (6 scenes, 36 files across original, exact copy, q85, q60, q30, 50% scale) through the full Stage 0–4 pipeline. |
| 18 | 2026-10-16 | Refined Reference Selection, Keeper Size Invariant & Calibrated Modes | Comprehensive revision of Stage 4 reference selection and keeper recommendations: (1) **Reference Selection**: Highest pixel count first; among equal pixels, prefer a lossless format (PNG/TIFF/BMP) only if it has no lossy twin of the same resolution; prefer canonical non-derived original filenames over copy suffixes (`__`, `_copy`, `_resave`), then smallest file size (never select a copy because it has more bytes per pixel); (2) **Keeper Size Invariant**: Recommended keeper file size must never exceed any `delete_safe` member's size by design unless that member is a lower-resolution copy (`keeper_sz <= m_sz` when `m_px >= keeper_px`), ensuring keeper is never larger than the smallest same-resolution member it replaces; (3) **Conservative Mode Rule**: Keeper is strictly the reference copy; `delete_safe` includes byte-identical copies of keeper plus members strictly worse (`m_px < k_px` or `m_quality < k_quality - 0.005`) and verified; nothing else is deleted automatically; (4) **Calibrated Mode Limits**: Balanced (`ssim_gray >= 0.98`, `ssim_color >= 0.97`, `pixel_ratio >= 0.60`, blur flag off, `min_saving_ratio 0.10`), Aggressive (`ssim_gray >= 0.95`, `ssim_color >= 0.94`, `pixel_ratio >= 0.25`, blur flag allowed, `min_saving_ratio 0.30`); (5) **Exact Duplicate Grouping**: Grouped members by `(md5, file_size)`; byte-identical copies of the keeper are always `delete_safe`; byte-identical copies of non-keepers evaluate anchor for review while marking duplicate copies `delete_safe`, preventing exact duplicates from stranding in review when keeper is a different file; (6) **Validation**: Verified across 15 unit tests in `tests/test_keeper.py` and 12 in `tests/test_quality.py` (124/124 repo-wide). Validated on `~/dedupe_test/demo` (6 scenes, 36 files): Conservative kept originals across 100% of clusters; Balanced kept q60-ish variants with original duplicate copies safely marked `delete_safe`; Aggressive kept compact q30 variants (24KB–30KB), strictly never exceeding original file size. |
| 19 | 2026-10-16 | JPEG Quality Estimation from Quantization Tables & Filename-Independent Reference Selection | Revised Stage 4 reference selection to eliminate reliance on filenames as a primary quality signal: (1) **JPEG Quality Estimation**: Extracted luminance quantization tables from JPEG markers (DQT `0xFFDB`) and matched them against standard IJG tables (Q 1–100) via vectorized $L_1$ norm in NumPy (`core/quality.py`, ~6.6 $\mu$s per image); persisted `jpeg_quality_est INTEGER` in SQLite `images` table; (2) **Reference Selection Hierarchy**: Strict priority: Highest pixel count (`-px`) $\to$ Lossless only if no lossy twin of same resolution $\to$ Highest estimated JPEG quality (`-jpeg_quality_est`) $\to$ Presence of EXIF capture time & camera make (`-exif_score`) $\to$ Earliest modified time (`mtime`) $\to$ Filename rule (fallback only: non-derived before copy suffixes, then path length) $\to$ Smallest file size $\to$ Deterministic tie-breaker (`path`, `id`); (3) **Unit Tests**: Added tests in `tests/test_quality.py` verifying: (a) q85 re-save with canonical name never beats a higher-quality (q95) original with copy-like name; (b) cluster of `IMG_0001.jpg`, `IMG_0001 (1).jpg`, `IMG_0001-edited.jpg` selects the highest-quality copy regardless of name; (c) shuffled filenames and list orders yield identical reference selection; (d) exact copy ties are 100% deterministic; (4) **Demo Verification**: Executed `bench/run_demo_comparison.py` on `~/dedupe_test/demo` across 6 scenes (36 files) comparing normal filenames vs randomized names (`img_<random>.jpg`). Results: 0/6 reference mismatches and 0/18 keeper mismatches across all modes (conservative, balanced, aggressive) — references and keepers are 100% identical. |
| 20 | 2026-10-16 | Source-Likelihood SSIM Centroid Scoring & Conservative Uncertain-Reference Freeze (`core/quality.py`, `core/keeper.py`) | Addressed flaw where estimated JPEG quality favoured a q85 re-save over its q75 original: (1) **Demoted `jpeg_quality_est`**: Demoted from primary selection rule to a tiebreaker evaluated after SSIM, EXIF, and mtime; (2) **Source-Likelihood SSIM Centroid**: For same-resolution lossy candidates, computed mean pairwise SSIM against all other same-resolution lossy peers on 1024px proxies (`core/quality.py`). The true source acts as the visual information centroid ($SSIM(O, \text{peers}) > SSIM(\text{resave}, \text{peers})$), reliably picking the true original over higher-quality re-saves; persisted in `images.source_likelihood`; (3) **Reference Selection Hierarchy**: Highest pixel count $\to$ Lossless format (if no lossy twin) $\to$ Highest source-likelihood score $\to$ EXIF presence (capture time + camera make) $\to$ Earliest mtime $\to$ `jpeg_quality_est` $\to$ Filename rule $\to$ Smallest file size $\to$ Deterministic tie-breaker; (4) **`reference_uncertain` & Conservative Freeze**: If top two source-likelihood scores differ by $< 0.002$ or EXIF/mtime contradicts SSIM ranking, flagged cluster as `reference_uncertain = 1` in `clusters`. In conservative mode, uncertain clusters freeze automatic deletion completely (`delete_safe = []`), routing all non-reference members to manual review; (5) **Unit Tests & Demo Verification**: Added tests in `tests/test_quality.py` for q75 vs q85/q60/q30, $\Delta < 0.002$ uncertainty trigger, and invariance under renaming/shuffling (131/131 tests passing repo-wide). Evaluated on `~/dedupe_test/demo` across 6 scenes (36 files) comparing normal names vs `img_<random>.jpg`: 0/6 reference mismatches, 0/18 keeper mismatches across all modes, and the true original won reference in 100% (6/6) of scenes. |
| 21 | 2026-10-16 | Byte-Identical Candidate Collapsing & Resolution Loss Protection (`core/quality.py`, `core/keeper.py`, `config.toml`) | Refined reference candidate grouping and keeper resolution safety: (1) **Byte-Identical Candidate Collapsing**: In `core/quality.py`, collapsed byte-identical members (same `md5` and `file_size`) into a single representative candidate before pairwise source-likelihood SSIM and before the uncertainty check. This ensures `reference_uncertain` compares only distinct visual candidates, preventing exact duplicates from triggering uncertainty against each other, while distributing scores to all identical copies; (2) **Resolution Loss Guard (`allow_resolution_loss`)**: Added `allow_resolution_loss = false` by default across conservative, balanced, and aggressive modes in `config.toml` and `DEFAULT_KEEPER_MODES`, with a user override flag in `choose_keeper` and CLI (`--allow-resolution-loss`). With the flag off, candidate keepers with lower pixel count than reference (`m_px < ref_px`) are excluded, ensuring aggressive mode keeps full-resolution files meeting limits (e.g. q30 or reference) rather than downscaling to `scale_50pct`; (3) **Unit Tests**: Added test suites verifying: (a) cluster of original, exact copy, and q60/q30 re-saves has `reference_uncertain = 0` and exact copy is `delete_safe` across all modes; (b) aggressive mode with flag off keeps a full-resolution file, while with flag on `scale_50pct` can become keeper (134/134 tests passing repo-wide); (4) **Demo Verification**: Evaluated on `~/dedupe_test/demo` (6 scenes, 36 files) across normal and random names: 0/6 reference mismatches, 0/18 keeper mismatches, with 6/6 true originals selected as references. |
| 22 | 2026-10-17 | Refined Uncertainty Rule, NULL Likelihood for Lone Members, and Exact Delete-Safe in Uncertain Conservative Mode (`core/quality.py`, `core/keeper.py`) | Addressed edge cases and precision rules for Stage 4 reference selection and keeper recommendations: (1) **NULL `source_likelihood` for Lone Members**: Assigned `NULL` (`None`) instead of `1.0` to `images.source_likelihood` when a member has no distinct same-resolution lossy peers in the cluster (e.g. `scale_50pct`), preventing false 100% source confidence; verified sorting and consumers safely treat `NULL` without crash; (2) **Preserve Byte-Identical Copies in Uncertain Conservative Mode**: When `mode == "conservative"` and `reference_uncertain = 1`, byte-identical copies of the keeper remain `delete_safe` (zero information lost regardless of which copy is source); only non-identical members route to `review_ids`; (3) **Refined Uncertainty Rule via Peer Majority Consistency**: A cluster is `reference_uncertain = 1` only if the top two distinct candidates have gap $\Delta < 0.002$ AND pairwise ordering is not consistent (i.e. top candidate does not win SSIM against the majority of other cluster peers) OR EXIF/mtime contradicts the SSIM ranking. If original beats re-save in pairwise peer comparisons and in mtime, the cluster is confident even with $\Delta < 0.002$; (4) **Byte-Identical Anchor Selection**: Documented and enforced anchor hierarchy: earliest modified time (`mtime`) $\to$ shortest path length $\to$ path $\to$ id, ensuring file renaming or shorter paths cannot change the anchor when mtimes differ; (5) **Unit Tests & Demo Verification**: Added unit tests in `tests/test_quality.py` and `tests/test_keeper.py` (138/138 tests passing repo-wide). Evaluated on `~/dedupe_test/demo` (6 scenes, 36 files) comparing normal vs randomized filenames (`img_<random>.jpg`): 0/6 reference mismatches, 0/18 keeper mismatches across all modes, 6/6 true originals selected, and peer head-to-head consistency confirmed all 6 demo scenes as confident (`reference_uncertain = 0`). |
| 23 | 2026-10-17 | Deletion Ledger, Safety Guards, Trash Management & CLI Commands (`core/ledger.py`, `core/cli.py`, `db/schema.py`, `config.toml`) | Implemented safe deletion execution, batch transaction lifecycle, and audit journaling: (1) **Trash-First Safe Deletion**: `core/ledger.py` is established as the sole authorized component for file removal in Argus. Permanent deletion is strictly prohibited during regular operations; files are relocated via `os.replace` / `shutil.move` to `~/.argus/trash/<batch_id>/` (or custom `--trash-dir`), preserving full relative directory hierarchy. Any filesystem failure leaves the source file in place and records the failure; (2) **Four-Stage Batch Lifecycle**: Operations strictly follow `plan` $\to$ `dry_run` $\to$ `confirm` $\to$ `execute`. `plan()` stages candidate entries into SQLite tables `ledger_batches` and `deletion_journal`. `dry_run()` simulates execution and validates all constraints. `execute()` enforces `confirm=True` before moving any files; (3) **Execution-Time Safety Checks**: Independent verification at execute time re-validates: file existence, file size, current MD5 vs recorded DB hash, non-keeper verification, prevention of whole-cluster deletion (ensuring at least one surviving member remains), and rejection of `review_flag=1`, `is_low_confidence=1`, or `quality_status='unalignable'` unless explicitly planned as a manual override (`is_manual=True`); (4) **Undo & Purge Controls**: Implemented `undo(batch_id)` and single-item `undo_file(journal_id)` to restore files to original paths without overwriting existing files (reporting conflicts safely). Implemented `purge(batch_id, confirm=True)` requiring explicit confirmation and strictly enforcing a minimum batch age (default 30 days via `purge_min_age_days`); (5) **CLI Integration & Comprehensive Tests**: Added CLI subcommands `argus plan`, `argus apply` (`--confirm`, `--dry-run`), `argus undo` (`--file`), and `argus history` (`--limit`). Added 13 unit tests in `tests/test_ledger.py` covering all safety and recovery invariants; entire repository test suite passes (151/151 tests in 28.7s). |
| 24 | 2026-10-17 | Scan Robustness, Format Support (.HEIC/.JPG case-insensitive), and Skipped/Decode-Failure Reporting (`core/stage0.py`, `core/stage1.py`, `sources/local_folder.py`, `core/cli.py`) | Investigated and resolved scan discovery gaps: (1) **Extension and Case-Insensitive Matching**: Scanner accepts `.arw`, `.bmp`, `.cr2`, `.cr3`, `.dng`, `.gif`, `.heic`, `.heif`, `.jpeg`, `.jpg`, `.nef`, `.orf`, `.pef`, `.png`, `.raf`, `.rw2`, `.tif`, `.tiff`, `.webp` using normalized `suffix.lower()`, matching uppercase extensions (`.JPG`, `.HEIC`, `.PNG`) identically to lowercase; CLI `argus scan` explicitly prints the accepted format list and case-insensitivity on startup; (2) **`pillow-heif` Registration & HEIC Scanning**: Confirmed `pillow_heif.register_heif_opener()` registration across `sources/local_folder.py`, `core/stage0.py`, and `core/cli.py`; added unit test verifying that `.heic` images scan, decode dimensions, hash MD5, and store format correctly; (3) **Explicit Skipped & Decode Failure Tracking**: Replaced silent error swallows in `get_metadata` and `_scan_and_sync_files`: corrupt or unopenable non-RAW files now raise descriptive `OSError` exceptions and are tracked in `Stage0Result.files_skipped` and `Stage0Result.skipped_reasons` (never silently ignored or erroneously admitted as discovered); empty 0-byte files and missing dimensions are similarly recorded; `cmd_scan` reports `Skipped: N` with filename and reason in the final scan summary; (4) **Rescan Unchanged Accounting**: Tracked `files_unchanged` in `Stage0Result` and displayed `Unchanged (in DB)` when rescanning folders so already-indexed files are not mistaken for discovery failures; (5) **Unit Tests**: Added `TestScanRobustnessAndFormats` in `tests/test_stage0.py` covering HEIC file scanning, folder with `.JPG, .jpeg, .PNG, .webp, .heic` and one corrupt file reporting 5 discovered and 1 skipped with reason, and CLI scan summary validation; all 155 repo-wide tests passing. |
| 25 | 2026-10-17 | False Trigger Resolution for `reference_uncertain` in Two-Member Clusters (`core/quality.py`, `tests/test_quality.py`) | Resolved false `reference_uncertain = 1` trigger observed in `~/dedupe_test/real.db` for two-member cluster containing `picsum_0032.jpg` (id 31) and `picsum_0032_lowq.jpg` (id 1): (1) **Root Cause Analysis**: Both same-resolution lossy members have identical pairwise SSIM (0.908447), producing $\Delta = 0.000 < 0.002$ (`is_close_score = True`). When sorting top candidates, `top_by_ssim` sorted solely by `source_likelihood`, arbitrarily assigning `c1 = picsum_0032_lowq` and `c2 = picsum_0032`. Because `picsum_0032` had an earlier mtime, `mtime2 < mtime1` fired as `is_mtime_contradiction = True`. Simultaneously, having 0 other peers (`peer_count = 0`) caused the majority vote check to default to `is_pairwise_consistent = False`, triggering `(is_close_score and not is_pairwise_consistent)`; (2) **Candidate Ordering via `key_fn`**: Sorted distinct top candidates using the reference priority `key_fn` rather than `source_likelihood` alone, ensuring `c1` is the candidate that wins the full reference priority hierarchy (earlier mtime, higher JPEG quality estimate, canonical non-derived filename); (3) **Two-Member Cluster Evidence Check**: When `peer_count == 0`, if source-likelihood gap $\ge 0.002$, `is_close_score = False` and the cluster is confident. When gap $< 0.002$, verified whether `c1` has substantive physical evidence of being the source over `c2` (earlier modified time, higher JPEG quality estimate, or better EXIF). If supported by metadata (as in `picsum_0032`, with JPEG quality 75 vs 56 and earlier mtime), `is_pairwise_consistent = True`, leaving `reference_uncertain = 0` (confident); if metadata is tied (e.g. `copy_a` vs `copy_b`), `reference_uncertain = 1`; (4) **Unit Tests**: Added tests in `tests/test_quality.py` for two-member clusters with higher source-likelihood, original + lowq re-save confidence, and contradicting mtime uncertainty; all 158 repo-wide tests passing. |

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
| Benchmark & Held-Out Test Evaluation | ✅ Done — 60/40 reproducible split (seed `42`), tuning on 60%, test on 40% (99.87% precision, 99.74% recall) |
| Stage 3 (Geometric Verification: SIFT + RANSAC) | ✅ Done (`core/verify.py`, re-evaluation of split members, inconclusive verdict for <N keypoints, SQLite keypoint cache, 99.87% precision, 99.74% recall) |
| Stage 4 Quality Scoring & Proxy Alignment | ✅ Done (`core/quality.py`, reference selection, bidirectional matrix caching in `pair_transforms`, SSIM, sharpness, CLI `quality`) |
| Stage 4 Keeper Selection & Safety Guards | ✅ Done (`core/keeper.py`, `models/duplicate_confidence.json`, CLI `keeper`) |
| Stage 4 (full-res deletion execution) | ✅ Done (`core/ledger.py`, trash-first relocation, execution safety checks, SQLite journaling, CLI `plan`, `apply`, `undo`, `history`) |
| API layer | 🔲 Next step |
| UI | 🔲 Not started |

