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

## 3. Final Performance on Held-Out Test Set Only

| Mode | Configured Threshold(s) | Precision | Variant Recall | Pairwise Recall | Clusters | Files Clustered |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **pHash only** | T=8 | 100.00% | 80.39% (1222/1520) | 65.54% | 83 | 1309 |
| **CLIP only** | cos=0.88 | 97.71% | 95.00% (1444/1520) | 90.55% | 80 | 1527 |
| **Combination** | T=6, cos=0.85 | **97.44%** | **100.00%** (1520/1520) | **100.00%** | 79 | 1600 |

## 4. Test Set Recall per Transformation Type (%)

| Transform | pHash only | CLIP only | Combination |
|:---|:---:|:---:|:---:|
| `brightness` | 100.0% | 100.0% | **100.0%** |
| `contrast` | 100.0% | 100.0% | **100.0%** |
| `crop` | 64.4% | 97.5% | **100.0%** |
| `exact_copy` | 100.0% | 100.0% | **100.0%** |
| `format_png` | 100.0% | 100.0% | **100.0%** |
| `format_webp` | 100.0% | 100.0% | **100.0%** |
| `jpeg_recompress` | 99.7% | 100.0% | **100.0%** |
| `resize` | 100.0% | 70.0% | **100.0%** |
| `rotate180` | 0.0% | 100.0% | **100.0%** |
| `rotate270` | 0.0% | 100.0% | **100.0%** |
| `rotate90` | 0.0% | 100.0% | **100.0%** |

## 5. Summary & Conclusions
- The thresholds chosen on the tuning set ($T=6, c=0.85$) generalized with outstanding fidelity to the held-out test set.
- Test set precision reached **97.44%** with **100.00%** variant recall (1520/1520 variants recalled across 80 unseen scenes).
- All transform variants achieved 97.5%–100% recall on unseen test images, confirming that Stage 0–2 achieves robust duplicate detection without overfitting.
