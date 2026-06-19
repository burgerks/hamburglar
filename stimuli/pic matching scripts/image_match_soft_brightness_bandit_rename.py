#!/usr/bin/env python3

from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pandas as pd
import statistics

# =========================
# USER SETTINGS
# =========================

INPUT_FOLDER = Path("/Users/burgerks/Desktop/allpics")
OUTPUT_FOLDER = Path("/Users/burgerks/Desktop/fixed_pics_soft_match")

# Recommended for photos/manuscript/web:
TARGET_WIDTH = 3000
TARGET_HEIGHT = 2000

DPI = 300

# Output options: "JPEG" or "PNG"
# JPEG is usually better for high-resolution photos because PNG files can be huge.
OUTPUT_FORMAT = "JPEG"
JPEG_QUALITY = 95

# Rename output files sequentially, e.g., bandit_001.jpg, bandit_002.jpg
OUTPUT_BASENAME = "bandit"
OUTPUT_NUMBER_PADDING = 3

# Supported input image types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# If True, subfolders are included.
RECURSIVE = False

# =========================
# BRIGHTNESS MATCHING SETTINGS
# =========================

# Use median luminance rather than mean luminance for the target.
# This is more robust when images have big dark regions, bright highlights, or uneven lighting.
TARGET_BRIGHTNESS_METRIC = "median"  # options: "median" or "mean"

# Partial correction: 1.00 = full correction, 0.00 = no correction.
# Recommended first try: 0.60
CORRECTION_STRENGTH = 0.60

# Clamp brightness changes so dark images do not become gray/noisy
# and bright images do not become washed out.
MIN_BRIGHTNESS_MULTIPLIER = 0.85
MAX_BRIGHTNESS_MULTIPLIER = 1.20

# Optional mild contrast correction after brightness matching.
# 1.00 = no change. Try 1.03 to 1.08 only if images look flat.
CONTRAST_MULTIPLIER = 1.00

# =========================
# FUNCTIONS
# =========================

def get_image_paths(folder: Path, recursive: bool = False):
    pattern = "**/*" if recursive else "*"
    return [
        p for p in folder.glob(pattern)
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith("._")
    ]


def crop_to_aspect(img: Image.Image, target_width: int, target_height: int):
    """
    Center-crop image to match target aspect ratio.
    No stretching.
    """
    w, h = img.size
    target_aspect = target_width / target_height
    image_aspect = w / h

    if image_aspect > target_aspect:
        # Image is too wide. Crop left/right.
        new_w = int(round(h * target_aspect))
        left = (w - new_w) // 2
        crop_box = (left, 0, left + new_w, h)
    else:
        # Image is too tall. Crop top/bottom.
        new_h = int(round(w / target_aspect))
        top = (h - new_h) // 2
        crop_box = (0, top, w, top + new_h)

    cropped = img.crop(crop_box)
    return cropped, crop_box


def luminance_values_0_255(img: Image.Image):
    """
    Rec.709/sRGB luminance values on a 0-255 scale.
    Formula:
    Y = 0.2126R + 0.7152G + 0.0722B
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)

    luminance = (
        0.2126 * arr[:, :, 0] +
        0.7152 * arr[:, :, 1] +
        0.0722 * arr[:, :, 2]
    )
    return luminance


def luminance_mean_0_255(img: Image.Image):
    return float(np.mean(luminance_values_0_255(img)))


def luminance_median_0_255(img: Image.Image):
    return float(np.median(luminance_values_0_255(img)))


def approximate_lstar_from_luminance(y_0_255: float):
    """
    Approximate CIELAB L* from display-referred luminance.
    This is a readable brightness metric, but the matching metric is luminance 0-255.
    """
    y = max(0.0, min(1.0, y_0_255 / 255.0))

    if y > 0.008856:
        return 116 * (y ** (1 / 3)) - 16
    else:
        return 903.3 * y


def make_unique_output_path(folder: Path, filename: str):
    """Avoid overwriting files with the same stem."""
    output_path = folder / filename
    if not output_path.exists():
        return output_path

    stem = output_path.stem
    suffix = output_path.suffix
    counter = 2
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# =========================
# MAIN SCRIPT
# =========================

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(get_image_paths(INPUT_FOLDER, recursive=RECURSIVE), key=lambda p: p.name.lower())

    if not image_paths:
        raise RuntimeError(f"No readable images found in: {INPUT_FOLDER}")

    print(f"Found {len(image_paths)} images.")

    # First pass:
    # crop + resize all images, then calculate brightness.
    prepped_images = []

    for path in image_paths:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            original_width, original_height = img.size

            cropped, crop_box = crop_to_aspect(
                img,
                TARGET_WIDTH,
                TARGET_HEIGHT
            )

            resized = cropped.resize(
                (TARGET_WIDTH, TARGET_HEIGHT),
                Image.Resampling.LANCZOS
            )

            mean_lum = luminance_mean_0_255(resized)
            median_lum = luminance_median_0_255(resized)

            prepped_images.append({
                "path": path,
                "filename": path.name,
                "original_width_px": original_width,
                "original_height_px": original_height,
                "crop_box": crop_box,
                "resized_image": resized,
                "pre_match_mean_luminance_0_255": mean_lum,
                "pre_match_median_luminance_0_255": median_lum,
            })

    # Target brightness:
    # Median-of-image-medians is usually more robust than mean matching.
    if TARGET_BRIGHTNESS_METRIC.lower() == "median":
        target_luminance = statistics.median(
            item["pre_match_median_luminance_0_255"]
            for item in prepped_images
        )
        target_metric_label = "batch median of image median luminance"
    elif TARGET_BRIGHTNESS_METRIC.lower() == "mean":
        target_luminance = statistics.median(
            item["pre_match_mean_luminance_0_255"]
            for item in prepped_images
        )
        target_metric_label = "batch median of image mean luminance"
    else:
        raise ValueError('TARGET_BRIGHTNESS_METRIC must be "median" or "mean"')

    target_lstar = approximate_lstar_from_luminance(target_luminance)

    print(f"Target dimensions: {TARGET_WIDTH} × {TARGET_HEIGHT} px")
    print(f"DPI metadata: {DPI}")
    print(f"Brightness target metric: {target_metric_label}")
    print(f"Target luminance: {target_luminance:.3f} / 255")
    print(f"Correction strength: {CORRECTION_STRENGTH:.2f}")
    print(f"Allowed brightness multiplier range: {MIN_BRIGHTNESS_MULTIPLIER:.2f} to {MAX_BRIGHTNESS_MULTIPLIER:.2f}")
    print(f"Approximate target L*: {target_lstar:.3f}")

    report_rows = []

    # Second pass:
    # Apply partial, clamped brightness correction and save images.
    for output_index, item in enumerate(prepped_images, start=1):
        img = item["resized_image"]

        if TARGET_BRIGHTNESS_METRIC.lower() == "median":
            pre_match_luminance_for_matching = item["pre_match_median_luminance_0_255"]
        else:
            pre_match_luminance_for_matching = item["pre_match_mean_luminance_0_255"]

        if pre_match_luminance_for_matching > 0:
            full_multiplier = target_luminance / pre_match_luminance_for_matching
        else:
            full_multiplier = 1.0

        # Partial correction:
        # 1.0 means no change. full_multiplier means full brightness match.
        unclamped_multiplier = 1.0 + CORRECTION_STRENGTH * (full_multiplier - 1.0)

        # Clamp so images are not overcorrected.
        brightness_multiplier = max(
            MIN_BRIGHTNESS_MULTIPLIER,
            min(MAX_BRIGHTNESS_MULTIPLIER, unclamped_multiplier)
        )
        was_clamped = brightness_multiplier != unclamped_multiplier

        matched = ImageEnhance.Brightness(img).enhance(brightness_multiplier)

        if CONTRAST_MULTIPLIER != 1.0:
            matched = ImageEnhance.Contrast(matched).enhance(CONTRAST_MULTIPLIER)

        post_mean = luminance_mean_0_255(matched)
        post_median = luminance_median_0_255(matched)

        if OUTPUT_FORMAT.upper() == "JPEG":
            output_suffix = ".jpg"
        elif OUTPUT_FORMAT.upper() == "PNG":
            output_suffix = ".png"
        else:
            raise ValueError('OUTPUT_FORMAT must be "JPEG" or "PNG"')

        output_name = f"{OUTPUT_BASENAME}_{output_index:0{OUTPUT_NUMBER_PADDING}d}{output_suffix}"
        output_path = OUTPUT_FOLDER / output_name

        matched = matched.convert("RGB")

        if OUTPUT_FORMAT.upper() == "JPEG":
            matched.save(
                output_path,
                format="JPEG",
                dpi=(DPI, DPI),
                quality=JPEG_QUALITY,
                subsampling=0,
                optimize=True
            )
        else:
            matched.save(
                output_path,
                format="PNG",
                dpi=(DPI, DPI)
            )

        left, top, right, bottom = item["crop_box"]

        report_rows.append({
            "source_file": item["filename"],
            "output_index": output_index,
            "output_file": output_path.name,
            "original_width_px": item["original_width_px"],
            "original_height_px": item["original_height_px"],
            "crop_left_px": left,
            "crop_top_px": top,
            "crop_right_px": right,
            "crop_bottom_px": bottom,
            "output_width_px": TARGET_WIDTH,
            "output_height_px": TARGET_HEIGHT,
            "dpi": DPI,
            "output_format": OUTPUT_FORMAT.upper(),
            "pre_match_mean_luminance_0_255": round(item["pre_match_mean_luminance_0_255"], 3),
            "pre_match_median_luminance_0_255": round(item["pre_match_median_luminance_0_255"], 3),
            "matching_metric_used": TARGET_BRIGHTNESS_METRIC.lower(),
            "pre_match_luminance_used_for_matching_0_255": round(pre_match_luminance_for_matching, 3),
            "target_luminance_0_255": round(target_luminance, 3),
            "full_brightness_multiplier_not_used": round(full_multiplier, 6),
            "unclamped_partial_multiplier": round(unclamped_multiplier, 6),
            "brightness_multiplier_applied": round(brightness_multiplier, 6),
            "brightness_multiplier_was_clamped": was_clamped,
            "correction_strength": CORRECTION_STRENGTH,
            "min_brightness_multiplier": MIN_BRIGHTNESS_MULTIPLIER,
            "max_brightness_multiplier": MAX_BRIGHTNESS_MULTIPLIER,
            "contrast_multiplier": CONTRAST_MULTIPLIER,
            "post_match_mean_luminance_0_255": round(post_mean, 3),
            "post_match_median_luminance_0_255": round(post_median, 3),
            "target_Lstar_approx": round(target_lstar, 3),
        })

    report = pd.DataFrame(report_rows)
    report_path = OUTPUT_FOLDER / "brightness_resize_report.csv"
    report.to_csv(report_path, index=False)

    readme_path = OUTPUT_FOLDER / "README_brightness_matching.txt"
    with open(readme_path, "w") as f:
        f.write("Image batch brightness and resize summary\n")
        f.write("=========================================\n\n")
        f.write(f"Images processed: {len(report_rows)}\n")
        f.write(f"Output dimensions: {TARGET_WIDTH} × {TARGET_HEIGHT} px\n")
        f.write(f"DPI metadata: {DPI} dpi\n")
        f.write(f"Output format: {OUTPUT_FORMAT.upper()}\n")
        f.write(f"Output naming: {OUTPUT_BASENAME}_### with {OUTPUT_NUMBER_PADDING} digits, sorted by input filename\n")
        if OUTPUT_FORMAT.upper() == "JPEG":
            f.write(f"JPEG quality: {JPEG_QUALITY}\n")
        f.write("\n")
        f.write("Brightness metric\n")
        f.write("-----------------\n")
        f.write("Rec.709/sRGB luminance on a 0-255 scale.\n")
        f.write("Formula: Y = 0.2126R + 0.7152G + 0.0722B\n\n")
        f.write(f"Target metric: {target_metric_label}\n")
        f.write(f"Target luminance: {target_luminance:.6f} / 255\n")
        f.write(f"Approximate target L*: {target_lstar:.3f}\n\n")
        f.write("Soft matching settings\n")
        f.write("----------------------\n")
        f.write(f"Correction strength: {CORRECTION_STRENGTH}\n")
        f.write(f"Minimum brightness multiplier: {MIN_BRIGHTNESS_MULTIPLIER}\n")
        f.write(f"Maximum brightness multiplier: {MAX_BRIGHTNESS_MULTIPLIER}\n")
        f.write(f"Contrast multiplier: {CONTRAST_MULTIPLIER}\n\n")
        f.write("For a future image using the same soft method:\n")
        f.write("1. Crop and resize it to the same dimensions.\n")
        f.write("2. Compute its luminance using the same target metric.\n")
        f.write("3. full_multiplier = target_luminance / image_luminance\n")
        f.write("4. partial_multiplier = 1 + correction_strength * (full_multiplier - 1)\n")
        f.write("5. applied_multiplier = clamp(partial_multiplier, min_multiplier, max_multiplier)\n")

    print("\nDone.")
    print(f"Processed images saved to: {OUTPUT_FOLDER}")
    print(f"CSV report saved to: {report_path}")
    print(f"README saved to: {readme_path}")


if __name__ == "__main__":
    main()
