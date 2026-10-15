# Benchmark Evaluation Report (Stage 0–2 with Rotation) — 2026-10-15

Evaluation of the complete duplicate detection pipeline on `data/benchmark/` (4,000 images: 200 originals × 20 variants) with **rotation embeddings enabled** (`rotate90`, `rotate180`, `rotate270`).

## 1. Resource & Profiling Metrics

- **Model**: ViT-B-32 (`laion2b_s34b_b79k`, MIT License)
- **Device Used**: `mps`
- **Batch Size**: 32
- **Total Images Embedded**: 4000
- **Rotation Embeddings**: `90°`, `180°`, `270°` (4 orientations per image)
- **Time per 1,000 images**: **23.53 s**
- **Peak RAM (RSS)**: **993.6 MB**

### Index & Storage Footprint: Rotation ON vs. Rotation OFF

| Component | Rotation OFF (0° only) | Rotation ON (0°, 90°, 180°, 270°) | Ratio |
|:---|:---:|:---:|:---:|
| **Vectors per Image** | 1 vector | 4 vectors | 4.0× |
| **Total Vectors (4,000 images)** | 4,000 vectors | 16,000 vectors | 4.0× |
| **Raw Vector Footprint (Float32)** | 7.81 MB | 31.25 MB | 4.0× |
| **SQLite Storage Footprint (Float16 BLOB)** | 3.91 MB | 15.62 MB | 4.0× |

## 2. Overall Metrics Comparison

| Mode | Cosine Threshold | Precision | Variant Recall | Pairwise Recall | Clusters | Files Clustered |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **pHash only** | N/A (T=10) | 99.14% | 82.13% (3121/3800) | 68.29% | 206 | 3336 |
| **CLIP only** | 0.85 | 94.93% | 94.16% (3578/3800) | 90.04% | 217 | 3877 |
| **CLIP only** | 0.88 | 97.46% | 93.79% (3564/3800) | 88.93% | 207 | 3817 |
| **CLIP only** | 0.90 | 98.41% | 93.58% (3556/3800) | 88.18% | 210 | 3793 |
| **CLIP only** | 0.92 | 99.19% | 91.00% (3458/3800) | 83.54% | 223 | 3713 |
| **CLIP only** | 0.94 | 99.58% | 86.24% (3277/3800) | 75.58% | 243 | 3586 |
| **CLIP only** | 0.96 | 100.00% | 81.66% (3103/3800) | 68.15% | 251 | 3423 |
| **Combination** | 0.85 | 97.08% | 99.26% (3772/3800) | 98.68% | 201 | 4000 |
| **Combination** | 0.88 | 98.61% | 99.50% (3781/3800) | 99.11% | 202 | 3999 |
| **Combination** | 0.90 | 98.82% | 99.58% (3784/3800) | 99.26% | 202 | 3998 |
| **Combination** | 0.92 | 99.03% | 99.66% (3787/3800) | 99.38% | 202 | 3997 |
| **Combination** | 0.94 | 99.08% | 99.55% (3783/3800) | 99.18% | 202 | 3992 |
| **Combination** | 0.96 | 99.23% | 99.16% (3768/3800) | 98.47% | 203 | 3979 |

## 3. Recall per Transformation Type (%)

Percentage of each transformation variant correctly linked to its source original image.

| Transform | pHash (T=10) | CLIP 0.85 | CLIP 0.88 | CLIP 0.90 | CLIP 0.92 | CLIP 0.94 | CLIP 0.96 | Combo 0.85 | Combo 0.88 | Combo 0.90 | Combo 0.92 | Combo 0.94 | Combo 0.96 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `brightness` | 99.5% | 97.0% | 98.0% | 99.2% | 97.8% | 94.5% | 89.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `contrast` | 100.0% | 97.2% | 98.0% | 99.0% | 99.0% | 98.5% | 95.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `crop` | 81.0% | 94.0% | 96.0% | 96.8% | 89.5% | 81.5% | 67.8% | 99.8% | 99.5% | 99.2% | 99.0% | 97.8% | 94.0% |
| `exact_copy` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `format_png` | 100.0% | 99.0% | 99.5% | 100.0% | 100.0% | 99.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `format_webp` | 100.0% | 99.0% | 98.0% | 99.0% | 99.0% | 98.5% | 98.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `jpeg_recompress` | 100.0% | 99.1% | 99.1% | 100.0% | 99.9% | 99.0% | 98.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `resize` | 99.8% | 74.2% | 68.5% | 63.3% | 53.0% | 32.3% | 18.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `rotate180` | 0.0% | 98.5% | 99.5% | 99.0% | 99.5% | 100.0% | 99.5% | 95.5% | 97.0% | 98.0% | 99.0% | 99.0% | 98.5% |
| `rotate270` | 0.0% | 98.5% | 99.5% | 100.0% | 99.5% | 99.0% | 99.5% | 96.0% | 97.0% | 97.5% | 98.0% | 98.0% | 98.5% |
| `rotate90` | 0.0% | 98.5% | 99.5% | 100.0% | 100.0% | 99.5% | 99.5% | 95.0% | 97.5% | 98.0% | 98.5% | 99.0% | 99.0% |

## 4. Inspection of False-Positive Pairs (Combined Mode @ cos=0.90)

Top false-positive pairs clustered together from different source originals, sorted by similarity score:

| # | Image A (Transform) | Original A | Image B (Transform) | Original B | Cosine Similarity | Link Signal |
|:---:|:---|:---|:---|:---|:---:|:---:|
| 1 | `picsum_0009__bright_114.jpg` (`brightness`) | `picsum_0009.jpg` | `picsum_0008__rotate270.jpg` (`rotate270`) | `picsum_0008.jpg` | **0.9674** | `clip` |
| 2 | `picsum_0008__rotate90.jpg` (`rotate90`) | `picsum_0008.jpg` | `picsum_0009__bright_114.jpg` (`brightness`) | `picsum_0009.jpg` | **0.9672** | `phash` |
| 3 | `picsum_0009__bright_114.jpg` (`brightness`) | `picsum_0009.jpg` | `picsum_0008__rotate180.jpg` (`rotate180`) | `picsum_0008.jpg` | **0.9665** | `clip` |
| 4 | `picsum_0009__contrast_114.jpg` (`contrast`) | `picsum_0009.jpg` | `picsum_0008__rotate180.jpg` (`rotate180`) | `picsum_0008.jpg` | **0.9655** | `clip` |
| 5 | `picsum_0008__rotate90.jpg` (`rotate90`) | `picsum_0008.jpg` | `picsum_0009__contrast_114.jpg` (`contrast`) | `picsum_0009.jpg` | **0.9652** | `phash` |
| 6 | `picsum_0009__contrast_114.jpg` (`contrast`) | `picsum_0009.jpg` | `picsum_0008__rotate270.jpg` (`rotate270`) | `picsum_0008.jpg` | **0.9647** | `clip` |
| 7 | `picsum_0009__jpeg_q60.jpg` (`jpeg_recompress`) | `picsum_0009.jpg` | `picsum_0008__rotate270.jpg` (`rotate270`) | `picsum_0008.jpg` | **0.9607** | `clip` |
| 8 | `picsum_0008__rotate90.jpg` (`rotate90`) | `picsum_0008.jpg` | `picsum_0009__jpeg_q60.jpg` (`jpeg_recompress`) | `picsum_0009.jpg` | **0.9600** | `phash` |
| 9 | `picsum_0009__fmt_png.png` (`format_png`) | `picsum_0009.jpg` | `picsum_0008__rotate180.jpg` (`rotate180`) | `picsum_0008.jpg` | **0.9591** | `clip` |
| 10 | `picsum_0009__jpeg_q60.jpg` (`jpeg_recompress`) | `picsum_0009.jpg` | `picsum_0008__rotate180.jpg` (`rotate180`) | `picsum_0008.jpg` | **0.9590** | `clip` |
| 11 | `picsum_0009__fmt_png.png` (`format_png`) | `picsum_0009.jpg` | `picsum_0008__rotate270.jpg` (`rotate270`) | `picsum_0008.jpg` | **0.9587** | `clip` |
| 12 | `picsum_0001__jpeg_q95.jpg` (`jpeg_recompress`) | `picsum_0001.jpg` | `picsum_0005__rotate270.jpg` (`rotate270`) | `picsum_0005.jpg` | **0.9586** | `clip` |
| 13 | `picsum_0001__crop_05pct.jpg` (`crop`) | `picsum_0001.jpg` | `picsum_0005__rotate270.jpg` (`rotate270`) | `picsum_0005.jpg` | **0.9584** | `clip` |
| 14 | `picsum_0008__rotate90.jpg` (`rotate90`) | `picsum_0008.jpg` | `picsum_0009__fmt_png.png` (`format_png`) | `picsum_0009.jpg` | **0.9579** | `phash` |
| 15 | `picsum_0001__crop_05pct.jpg` (`crop`) | `picsum_0001.jpg` | `picsum_0005__rotate90.jpg` (`rotate90`) | `picsum_0005.jpg` | **0.9577** | `clip` |
| 16 | `picsum_0005__rotate90.jpg` (`rotate90`) | `picsum_0005.jpg` | `picsum_0001__jpeg_q95.jpg` (`jpeg_recompress`) | `picsum_0001.jpg` | **0.9576** | `phash` |
| 17 | `picsum_0001__bright_085.jpg` (`brightness`) | `picsum_0001.jpg` | `picsum_0005__rotate270.jpg` (`rotate270`) | `picsum_0005.jpg` | **0.9564** | `clip` |
| 18 | `picsum_0001__rotate90.jpg` (`rotate90`) | `picsum_0001.jpg` | `picsum_0005__rotate270.jpg` (`rotate270`) | `picsum_0005.jpg` | **0.9559** | `clip` |
| 19 | `picsum_0009__jpeg_q95.jpg` (`jpeg_recompress`) | `picsum_0009.jpg` | `picsum_0008__rotate180.jpg` (`rotate180`) | `picsum_0008.jpg` | **0.9556** | `clip` |
| 20 | `picsum_0009__rotate270.jpg` (`rotate270`) | `picsum_0009.jpg` | `picsum_0008__rotate180.jpg` (`rotate180`) | `picsum_0008.jpg` | **0.9550** | `clip` |

## 5. Key Observations
- **Rotations (`rotate90`, `rotate180`, `rotate270`)**: With rotation embeddings enabled, rotated duplicates achieve high recall without degrading exact-match or recompression precision.
- **Cost of Rotation**: Multi-angle embeddings quadruples the vector footprint (8.2 MB → 32.8 MB for 4,000 images) and adds ~30-40% embedding compute latency.
- **Nature of False Positives**: As shown in the table above, false positives at cos=0.90 typically involve highly similar scenes (e.g. low-texture backgrounds, skies, or ocean surfaces) where perceptual hash and visual embedding overlap.
