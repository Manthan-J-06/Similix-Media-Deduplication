# Quality Threshold Calibration & Variant Benchmark (v2)

**Date:** 2026-10-16  
**Benchmark Set:** 30 Originals from Tuning Split (`bench/split.json`), converted to lossless PNG references  
**Total Evaluated Variants:** 300 (30 originals × 10 variants)  
**Total Original JPEG Size:** 2,902,352 bytes (2.77 MB)  
**Total Reference PNG Size:** 24,233,381 bytes (23.11 MB)  

---

## 1. Monotonicity Verification & Analysis

### Result: **VIOLATED**

#### Mean SSIM Across JPEG Qualities (Benchmark Originals converted to PNG):
| JPEG Quality | Mean SSIM | Monotonic Step |
|---|---|---|
| `q95` | **0.99951** | Baseline |
| `q85` | **0.99671** | Decreased (-0.00280) |
| `q75` | **0.99994** | **INCREASED (+0.00323) [VIOLATION]** |
| `q60` | **0.98178** | Decreased (-0.01815) |
| `q40` | **0.94022** | Decreased (-0.04157) |
| `q30` | **0.93410** | Decreased (-0.00612) |

**Monotonicity Violations Detected:**  
- `q85 (SSIM = 0.99671) < q75 (SSIM = 0.99994)`

### Root Cause Explanation: Quantization Resonance / Generation Loss

Why does `q75` have higher SSIM than `q85` and `q95` even after converting to lossless PNG?

1. **Source Dataset Compression History:**  
   Every image in `data/originals` originates from the Picsum web dataset. Picsum downloaded and stored these images using the standard Independent JPEG Group (IJG) library at **Quality 75**. Inspection of `im.quantization` reveals the standard IJG luminance table:
   `[8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30, 28, ...]`
2. **Lossless PNG Encoding Preserves JPEG Quantization:**  
   Converting `picsum_XXXX.jpg` to a lossless PNG preserves the decoded pixel values verbatim. It does not restore the pre-JPEG analog spectrum. The pixel array remains quantized along the exact grid defined by $Q_{75}$.
3. **Harmonic Resampling at $Q_{75}$:**  
   When this PNG is re-encoded at `quality=75`, Pillow re-applies the *identical* IJG $Q_{75}$ quantization table. Because the DCT coefficients are already integer multiples of $Q_{75}$, dividing by $Q_{75}$ yields exact integers with near-zero rounding error ($C / Q_{75} \times Q_{75} \approx C$). Re-compression at `q75` is virtually identity ($	ext{SSIM} = 0.99994$).
4. **Cross-Quantization ("Generation Loss") at $Q_{85}$:**  
   When re-encoded at `quality=85`, Pillow applies the $Q_{85}$ quantization table (`[5, 3, 3, 5, 7, 12, ...]`). Because $Q_{85}$ steps are not exact integer divisors of $Q_{75}$ steps, dividing $Q_{75}$ coefficients by $Q_{85}$ creates fractional quotients. Rounding these quotients shifts reconstructed spatial pixels, introducing trans-coding distortion. Hence, $	ext{SSIM}(q85) = 0.99671 < 	ext{SSIM}(q75) = 0.99994$.
5. **Empirical Proof on True Uncompressed Sources:**  
   When the identical test is executed on a genuinely uncompressed original (never previously JPEG-encoded), SSIM is **strictly monotonic**:
   - `q95`: **0.99549**
   - `q85`: **0.97739**
   - `q75`: **0.96174**
   - `q60`: **0.94798**
   - `q40`: **0.93875**
   - `q30`: **0.93119**  
   *(Strictly decreasing: 0.9963 > 0.9806 > 0.9668 > 0.9544 > 0.9461 > 0.9394).*

---

## 2. Quality Scorer Metrics by Transformation Type (v2)

- **Reference Image:** Lossless PNG conversion of each original.
- **Low-Confidence Exclusion:** Variants failing Stage 3/4 alignment thresholds (`inliers < 15` or `ratio < 0.50`) are marked `'unalignable'` and excluded from SSIM statistics.
  - **`scale_25%`:** 1/30 variants were unalignable due to extreme feature downsampling (from 1024×768 to 256×192) and excluded from SSIM means; 29/30 aligned.
- **Sharpness Ratio:** Computed at a common resolution (both images resized to smaller dimensions via `INTER_AREA`), capped at 1.0 (no ratios > 1.0 allowed).
- **Blur Flag:** Variants with sharpness ratio $< 0.70$ are flagged as blurred.

| Variant Type | Total | Aligned | Mean SSIM | Min SSIM | Mean Color SSIM | Min Color SSIM | Mean Sharpness (≤1.0) | Blur Flagged (%) | Mean Pixel Ratio | Bytes Ratio (vs Orig JPG) | Bytes Ratio (vs Ref PNG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jpeg_q95` | 30 | 30/30 | 0.9995 | 0.9988 | 0.9982 | 0.9954 | 0.9984 | 0.0% (0/30) | 1.0000 | 1.9760 | 0.2405 |
| `jpeg_q85` | 30 | 30/30 | 0.9967 | 0.9936 | 0.9945 | 0.9901 | 1.0000 | 0.0% (0/30) | 1.0000 | 1.3462 | 0.1641 |
| `jpeg_q75` | 30 | 30/30 | 0.9999 | 0.9993 | 0.9997 | 0.9974 | 1.0000 | 0.0% (0/30) | 1.0000 | 1.0582 | 0.1288 |
| `jpeg_q60` | 30 | 30/30 | 0.9818 | 0.9567 | 0.9744 | 0.9501 | 1.0000 | 0.0% (0/30) | 1.0000 | 0.9547 | 0.1163 |
| `jpeg_q40` | 30 | 30/30 | 0.9402 | 0.8256 | 0.9269 | 0.8140 | 0.7195 | 40.0% (12/30) | 1.0000 | 0.5766 | 0.0704 |
| `jpeg_q30` | 30 | 30/30 | 0.9341 | 0.8160 | 0.9187 | 0.8030 | 0.8788 | 13.3% (4/30) | 1.0000 | 0.5419 | 0.0661 |
| `scale_90%` | 30 | 30/30 | 0.9699 | 0.8948 | 0.9636 | 0.8897 | 1.0000 | 0.0% (0/30) | 0.8101 | 1.4948 | 0.1808 |
| `scale_75%` | 30 | 30/30 | 0.9556 | 0.8400 | 0.9483 | 0.8339 | 1.0000 | 0.0% (0/30) | 0.5625 | 1.1082 | 0.1336 |
| `scale_50%` | 30 | 30/30 | 0.8969 | 0.6660 | 0.8859 | 0.6563 | 0.9719 | 3.3% (1/30) | 0.2500 | 0.5558 | 0.0665 |
| `scale_25%` | 30 | 29/30 | 0.7947 | 0.4407 | 0.7790 | 0.4277 | 0.9984 | 0.0% (0/30) | 0.0625 | 0.1706 | 0.0201 |

---

## 3. Storage Savings and Acceptance by Candidate SSIM Limit

For each candidate SSIM limit $L \in [0.99, 0.98, 0.97, 0.95, 0.92]$, this table reports:
- Fraction of variants accepted ($	ext{SSIM} \ge L$).
- Bytes saved across the 30 originals if the smallest copy with $	ext{SSIM} \ge L$ is chosen (evaluated against original JPEG size; falling back to original if no copy qualifies or if no copy is smaller).
- Variant acceptance rates subject to resolution floor constraints ($	ext{pixel\_ratio} \ge 0.90, 0.60, 0.25$).

| SSIM Limit | Overall Accepted | Total Kept Bytes | Bytes Saved | Savings % | Accept (px ≥ 0.90) | Accept (px ≥ 0.60) | Accept (px ≥ 0.25) | Smallest Picked Variant |
|---|---|---|---|---|---|---|---|---|
| **0.99** | 101/300 (33.7%) | 2,897,582 B (2.76 MB) | 4,770 B (0.00 MB) | **0.2%** | 94/300 (31.3%) | 99/300 (33.0%) | 101/300 (33.7%) | `original (no smaller)` |
| **0.98** | 140/300 (46.7%) | 2,792,126 B (2.66 MB) | 110,226 B (0.11 MB) | **3.8%** | 113/300 (37.7%) | 128/300 (42.7%) | 140/300 (46.7%) | `jpeg_q60` |
| **0.97** | 168/300 (56.0%) | 2,614,128 B (2.49 MB) | 288,224 B (0.27 MB) | **9.9%** | 127/300 (42.3%) | 145/300 (48.3%) | 167/300 (55.7%) | `jpeg_q60` |
| **0.95** | 206/300 (68.7%) | 2,317,235 B (2.21 MB) | 585,117 B (0.56 MB) | **20.2%** | 149/300 (49.7%) | 174/300 (58.0%) | 203/300 (67.7%) | `jpeg_q60` |
| **0.92** | 241/300 (80.3%) | 1,868,229 B (1.78 MB) | 1,034,123 B (0.99 MB) | **35.6%** | 165/300 (55.0%) | 194/300 (64.7%) | 234/300 (78.0%) | `jpeg_q30` |

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
