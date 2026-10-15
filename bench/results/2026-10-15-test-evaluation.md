# Held-Out Test Set Evaluation Report — 2026-10-15

## 1. Benchmark Split Methodology

- **Split File**: `bench/split.json` (reproducible)
- **Random Seed**: `42`
- **Tuning Set**: **60%** (120 originals → 2400 images)
- **Held-Out Test Set**: **40%** (80 originals → 1600 images)
- **Data Leakage Guarantee**: Images are split by original source file; all variants of an original stay together strictly in either tuning or test.

## 2. Thresholds Chosen on Tuning Set Only

- **pHash only**: Hamming Threshold $T = 8$ (Tuning F1: 89.70)
- **CLIP only**: Cosine Threshold $c = 0.88$ (Tuning F1: 96.14)
- **Combination**: pHash $T = 6$, CLIP Cosine $c = 0.85$ (Tuning F1: 100.00)

## 3. Performance on Held-Out Test Set (Before & After Stage 3)

| Pipeline Stage / Mode | Threshold(s) | Precision | Variant Recall | Pairwise Recall | Clusters | Clustered Files |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **pHash only** | T=8 | 100.00% | 80.39% (1222/1520) | 65.54% | 83 | 1309 |
| **CLIP only** | cos=0.88 | 97.71% | 95.00% (1444/1520) | 90.55% | 80 | 1527 |
| **Combo (Stage 0–2)** | T=6, cos=0.85 | 97.44% | 100.00% (1520/1520) | 100.00% | 79 | 1600 |
| **Combo + Stage 3 (SIFT RANSAC)** | + SIFT/Affine (inl>=20, r>=0.65) | **99.60%** | **99.74%** (1516/1520) | **99.51%** | 80 | 1599 |

## 4. Stage 3 Verification Computational Overhead

- **Geometric Verification Time**: **29.27s** total for 1521 candidate pairs
- **Throughput / Latency**: **18.29s per 1,000 images**
- **Candidate Pairs Verified**: 1372 / 1521
- **Inconclusive Candidate Pairs**: 129 (keypoints < min_keypoints, kept in cluster)
- **Low Confidence Members**: **171** (excluded from auto-delete recommendations)
- **Impostor Members Split**: 20
- **New Clusters Formed**: 1 (split members successfully re-grouped)
- **Clusters Disbanded**: 0

## 5. Test Set Recall per Transformation Type (%)

| Transform | pHash only | CLIP only | Combo (Stage 0–2) | Combo + Stage 3 |
|:---|:---:|:---:|:---:|:---:|
| `brightness` | 100.0% | 100.0% | 100.0% | **100.0%** |
| `contrast` | 100.0% | 100.0% | 100.0% | **99.4%** |
| `crop` | 64.4% | 97.5% | 100.0% | **100.0%** |
| `exact_copy` | 100.0% | 100.0% | 100.0% | **100.0%** |
| `format_png` | 100.0% | 100.0% | 100.0% | **100.0%** |
| `format_webp` | 100.0% | 100.0% | 100.0% | **100.0%** |
| `jpeg_recompress` | 99.7% | 100.0% | 100.0% | **100.0%** |
| `resize` | 100.0% | 70.0% | 100.0% | **98.8%** |
| `rotate180` | 0.0% | 100.0% | 100.0% | **100.0%** |
| `rotate270` | 0.0% | 100.0% | 100.0% | **100.0%** |
| `rotate90` | 0.0% | 100.0% | 100.0% | **100.0%** |

## 6. Summary & Conclusions
- **Precision**: Stage 3 geometric keypoint matching preserves **99.60%** precision on the held-out test set.
- **Overhead**: Adding Stage 3 keypoint verification on 512px proxies takes only **18.29s per 1,000 images**.
- **Robustness**: Rotated duplicates (90°, 180°, 270°) and scale/crop variants are robustly verified while rejecting cross-scene false positive pairs.
