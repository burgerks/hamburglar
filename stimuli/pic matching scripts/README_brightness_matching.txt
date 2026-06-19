Image batch brightness and resize summary
=========================================

Images processed: 222
Output dimensions: 3000 × 2000 px
DPI metadata: 300 dpi
Output format: JPEG
Output naming: bandit_### with 3 digits, sorted by input filename
JPEG quality: 95

Brightness metric
-----------------
Rec.709/sRGB luminance on a 0-255 scale.
Formula: Y = 0.2126R + 0.7152G + 0.0722B

Target metric: batch median of image median luminance
Target luminance: 97.131996 / 255
Approximate target L*: 68.088

Soft matching settings
----------------------
Correction strength: 0.6
Minimum brightness multiplier: 0.85
Maximum brightness multiplier: 1.2
Contrast multiplier: 1.0

For a future image using the same soft method:
1. Crop and resize it to the same dimensions.
2. Compute its luminance using the same target metric.
3. full_multiplier = target_luminance / image_luminance
4. partial_multiplier = 1 + correction_strength * (full_multiplier - 1)
5. applied_multiplier = clamp(partial_multiplier, min_multiplier, max_multiplier)
