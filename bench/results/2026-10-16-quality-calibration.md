# Quality Threshold Calibration & Variant Benchmark

**Date:** 2026-10-16  
**Benchmark Set:** 30 Originals from Tuning Split (`bench/split.json`)  
**Total Evaluated Variants:** 300 (30 originals × 10 variants)  
**Total Original Size:** 2,902,352 bytes (2.77 MB)  

---

## 1. Rotation Check: Exact Rotation vs. Interpolated Warping

When verifying 90°/180°/270° rotated variants against their unrotated reference:
- **Interpolated affine warp (`cv2.warpAffine`):** bilinear/bicubic interpolation resamples the pixel grid, introducing sub-pixel smoothing artifacts that artificially depress SSIM.
- **Exact orthogonal rotation (`cv2.rotate`):** pure index permutation with zero resampling interpolation error.

### Result on `A_rot90.jpg` vs. `A_conv.png`:
| Method | SSIM (Grayscale) | SSIM (Color) | Verdict |
|---|---|---|---|
| **Before:** Interpolated `warpAffine` | 0.9244 | 0.9212 | Depressed by interpolation smoothing |
| **After:** Exact `cv2.rotate` | **0.9978** | **0.9950** | Matches ~0.99+ expectation |

> **Decision:** Because SSIM rose from 0.9244 to 0.9978 (well exceeding 0.99), exact orthogonal rotation is now the default path in `core/quality.py` (`detect_orthogonal_rotation`).

---

## 2. Quality Scorer Metrics by Transformation Type

Evaluated across 30 benchmark originals with 6 JPEG recompression levels (`q95` to `q30`) and 4 downscaling ratios (`90%` to `25%`):

| Variant Type | Count | Mean SSIM | Min SSIM | Mean Color SSIM | Min Color SSIM | Mean Sharpness | Min Sharpness | Mean Pixel Ratio | Mean Bytes Ratio | Min Bytes Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| `jpeg_q95` | 30 | 0.9995 | 0.9988 | 0.9982 | 0.9954 | 1.0010 | 0.9851 | 1.0000 | 1.9760 | 1.8648 |
| `jpeg_q85` | 30 | 0.9967 | 0.9936 | 0.9945 | 0.9901 | 1.1605 | 1.0787 | 1.0000 | 1.3462 | 1.2640 |
| `jpeg_q75` | 30 | 0.9999 | 0.9993 | 0.9997 | 0.9974 | 1.0007 | 0.9998 | 1.0000 | 1.0582 | 1.0193 |
| `jpeg_q60` | 30 | 0.9818 | 0.9567 | 0.9744 | 0.9501 | 1.6430 | 1.3300 | 1.0000 | 0.9547 | 0.9007 |
| `jpeg_q40` | 30 | 0.9402 | 0.8256 | 0.9269 | 0.8140 | 0.7200 | 0.2989 | 1.0000 | 0.5766 | 0.3889 |
| `jpeg_q30` | 30 | 0.9341 | 0.8160 | 0.9187 | 0.8030 | 0.8988 | 0.4298 | 1.0000 | 0.5419 | 0.3762 |
| `scale_90%` | 30 | 0.9699 | 0.8948 | 0.9636 | 0.8897 | 1.0724 | 0.8682 | 0.8101 | 1.4948 | 1.4020 |
| `scale_75%` | 30 | 0.9556 | 0.8400 | 0.9483 | 0.8339 | 1.4193 | 0.8909 | 0.5625 | 1.1082 | 0.9803 |
| `scale_50%` | 30 | 0.8969 | 0.6660 | 0.8859 | 0.6563 | 2.0736 | 0.7066 | 0.2500 | 0.5558 | 0.4265 |
| `scale_25%` | 30 | 0.7682 | 0.0000 | 0.7531 | 0.0000 | 3.8072 | 0.5087 | 0.0625 | 0.1706 | 0.1035 |

### Key Metric Observations:
1. **JPEG Recompressions:**
   - `q95` and `q85` yield near-lossless SSIM (mean 0.9995 and 0.9967) with sharpness ratio near 1.00.
   - `q75` averages 0.9999 SSIM with minimum 0.9993.
   - `q60` through `q30` drop SSIM progressively from 0.9818 down to 0.9341 while reducing bytes ratio down to ~0.54x.
2. **Downscalings:**
   - `scale_90%` retains high fidelity (mean SSIM 0.9699).
   - `scale_75%` achieves 0.9556 mean SSIM with 1.11x bytes ratio (re-encoded at q90).
   - `scale_50%` achieves 0.8969 mean SSIM with 0.56x bytes ratio.
   - `scale_25%` drops to 0.7682 mean SSIM with 0.17x bytes ratio (1 image at 25% was unalignable due to severe feature downsampling).
3. **Sharpness Ratio:**
   - Downscaled images exhibit higher high-frequency Laplacian variance on proxy scaling due to edge sharpening from Lanczos downsampling.

---

## 3. Storage Savings and Acceptance by Candidate SSIM Limit

For each candidate SSIM threshold $L \in [0.99, 0.98, 0.97, 0.95, 0.92]$, this table shows:
- What fraction of member variants satisfy $\text{SSIM} \ge L$.
- How many bytes would be saved across the 30 originals if the smallest copy with $\text{SSIM} \ge L$ were chosen (falling back to the original if no copy qualifies or if no copy is smaller).

| SSIM Limit | Accepted Variants | Fraction Accepted | Total Kept Bytes | Bytes Saved | Savings % | Most Common Picked Variant |
|---|---|---|---|---|---|---|
| **0.99** | 101/300 | 33.7% | 2,897,582 B (2.76 MB) | 4,770 B (0.00 MB) | **0.2%** | `original (no smaller)` |
| **0.98** | 140/300 | 46.7% | 2,792,126 B (2.66 MB) | 110,226 B (0.11 MB) | **3.8%** | `jpeg_q60` |
| **0.97** | 168/300 | 56.0% | 2,614,128 B (2.49 MB) | 288,224 B (0.27 MB) | **9.9%** | `jpeg_q60` |
| **0.95** | 206/300 | 68.7% | 2,317,235 B (2.21 MB) | 585,117 B (0.56 MB) | **20.2%** | `jpeg_q60` |
| **0.92** | 241/300 | 80.3% | 1,868,229 B (1.78 MB) | 1,034,123 B (0.99 MB) | **35.6%** | `jpeg_q30` |

---

## 4. Variant Acceptance Breakdown per SSIM Limit

| Variant Type | SSIM ≥ 0.99 | SSIM ≥ 0.98 | SSIM ≥ 0.97 | SSIM ≥ 0.95 | SSIM ≥ 0.92 |
|---|---|---|---|---|---|
| `jpeg_q95` | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) |
| `jpeg_q85` | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) |
| `jpeg_q75` | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) |
| `jpeg_q60` | 4/30 (13%) | 21/30 (70%) | 27/30 (90%) | 30/30 (100%) | 30/30 (100%) |
| `jpeg_q40` | 0/30 (0%) | 1/30 (3%) | 5/30 (17%) | 16/30 (53%) | 24/30 (80%) |
| `jpeg_q30` | 0/30 (0%) | 1/30 (3%) | 5/30 (17%) | 13/30 (43%) | 21/30 (70%) |
| `scale_90%` | 5/30 (17%) | 15/30 (50%) | 18/30 (60%) | 25/30 (83%) | 29/30 (97%) |
| `scale_75%` | 2/30 (7%) | 10/30 (33%) | 15/30 (50%) | 19/30 (63%) | 25/30 (83%) |
| `scale_50%` | 0/30 (0%) | 2/30 (7%) | 7/30 (23%) | 10/30 (33%) | 15/30 (50%) |
| `scale_25%` | 0/30 (0%) | 0/30 (0%) | 1/30 (3%) | 3/30 (10%) | 7/30 (23%) |

---

*Generated automatically by `bench/quality_calibration.py`.*
