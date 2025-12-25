"""augment_dataset.py

Offline pipeline that — **for each text region** —
1. computes four mask‑based augmented variants **and** keeps the original;  
2. writes  the variants to disk using the naming convention:
    {text_region_stem}_{variant}.png

"""

from pathlib import Path
import sys
import os
import numpy as np
import cv2
from skimage.measure import label, regionprops
import shutil

# add external library path
sys.path.append("/content/drive/MyDrive/Colab Notebooks/icdar_fest_14_02_2025")
import ani  # make sure this module is on the sys.path


def estimate_binary_height(binary, th_low, th_high):
    labeled = label(binary)
    props = regionprops(labeled)
    heights = [
        p.bbox[2] - p.bbox[0]
        for p in props
        if th_low <= (p.bbox[2] - p.bbox[0]) <= th_high
    ]
    if not heights:
        return None
    mu = np.mean(heights)
    sigma = np.std(heights)
    return int(mu - 0.5 * sigma), int(mu + 0.5 * sigma)

def load_and_preprocess_image(image_path):
    # read in grayscale and binarize with Otsu
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_bin = cv2.bitwise_not(bin_img)
    return gray, bin_img, inv_bin

def filter_regions(image, scale, eta, thetas):
    img = image.astype(np.float32) / 255.0
    H, W = img.shape

    max_response = np.full((H, W), -np.inf, dtype=np.float32)
    for theta in thetas:
        resp = ani.anigauss(img, scale, eta * scale, theta, 0, 0)
        np.maximum(max_response, resp, out=max_response)

    return max_response


# ──-------------------------------------------------------------------------
# 🔧  Parameters
# ────────────────────────────────────────────────────────────────────────────
THETAS     = [0, 90]
SCALE      = 1
ETA_FACTOR = 1

# suffixes for augmented variants (empty string → original)
VARIANTS = {
    "_inverse":                    lambda gray,  bin_img, bg: 255 - gray,
    "_masked_inv_gray":    lambda gray, bin_img, bg: cv2.bitwise_and(255-gray, 255-gray, mask=bg),
    "_inv_masked_gray":     lambda gray, bin_img, bg: 255 - cv2.bitwise_and(gray, gray, mask=bg),
    "_inv_masked_bin":      lambda gray, bin_img, bg: 255 - cv2.bitwise_and(bin_img, bin_img, mask=bg),
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ──-------------------------------------------------------------------------
# 🚀  Helpers
# ────────────────────────────────────────────────────────────────────────────


def process_page(img_path: Path, dataset_dir: Path, aug_dataset_dir: Path):
    """
    Mirrors the folder structure of dataset_dir inside aug_dataset_dir.
    Saves:
        - the original image (copied)
        - all augmented variants
    """

    # ---------------------------------------------------------
    # 1. Compute relative path, e.g. split/script_mode/filename
    # ---------------------------------------------------------
    rel_path = img_path.relative_to(dataset_dir)
    split = rel_path.parts[0]
    script_mode = rel_path.parts[1]
    filename = rel_path.name

    # ---------------------------------------------------------
    # 2. Build output directory
    # ---------------------------------------------------------
    out_dir = aug_dataset_dir / split / script_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Copy the original image to the aug dataset directory
    # ---------------------------------------------------------
    dst_original = out_dir / filename
    shutil.copy2(img_path, dst_original)

    # ---------------------------------------------------------
    # 4. Load & preprocess
    # ---------------------------------------------------------
    gray, bin_img, inv_bin = load_and_preprocess_image(str(img_path))

    ths_low  = bin_img.shape[0] * 0.001
    ths_high = bin_img.shape[0] * 0.1
    avr_h    = estimate_binary_height(inv_bin, ths_low, ths_high)[0]
    eta      = ETA_FACTOR * avr_h
    bg_mask  = (filter_regions(bin_img, SCALE, eta, THETAS) < 0.99).astype(np.uint8) * 255

    # ---------------------------------------------------------
    # 5. Generate & save each variant
    # ---------------------------------------------------------
    for suffix, fn in VARIANTS.items():

        variant_img = fn(gray, bin_img, bg_mask)

        out_name = f"{img_path.stem}{suffix}.jpg"
        cv2.imwrite(str(out_dir / out_name), variant_img)


# ──-------------------------------------------------------------------------
# 🏃‍♀️  Main driver
# ────────────────────────────────────────────────────────────────────────────

def augment_ds(ds_dir: Path, aug_ds_dir: Path):
    for split in ("train", "val"):
        for script_mode in ["french", "german"]:
            images_dir = ds_dir / split / script_mode
            if not images_dir.exists():
                print(f"⏭  {images_dir} not found – skipping.")
                continue
            pages = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
            print(f"▶  {split:<10} / {script_mode:<10} | {len(pages):6d} pages")

            for p in pages:
                try:
                    process_page(p, ds_dir, aug_ds_dir)
                except Exception as e:
                    print(f"[WARN] {p}: {e}")

def main():
    ds_dir = Path("ASC_dataset_v3")
    aug_ds_dir = Path("ASC_dataset_v3_augmented")
    augment_ds(ds_dir, aug_ds_dir)


if __name__ == "__main__":
    main()
