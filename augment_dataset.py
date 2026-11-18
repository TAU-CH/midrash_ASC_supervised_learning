"""augment_dataset.py

Offline pipeline that, for each page image:
1. computes four mask-based augmented variants and keeps the original;
2. writes all outputs as JPG files into a sibling augmented directory.

By default the script augments both:
- ICDAR2017_CLaMM_Training -> ICDAR2017_CLaMM_Training_augmented
- ICDAR2017_CLaMM_evaluation_set -> ICDAR2017_CLaMM_evaluation_set_augmented
"""

from pathlib import Path
import sys
import numpy as np
import cv2
from skimage.measure import label, regionprops

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
MIN_TEXT_HEIGHT = 1

# suffixes for augmented variants (empty string → original)
VARIANTS = {
    "_inverse":                    lambda gray,  bin_img, bg: 255 - gray,
    "_masked_inv_gray":    lambda gray, bin_img, bg: cv2.bitwise_and(255-gray, 255-gray, mask=bg),
    "_inv_masked_gray":     lambda gray, bin_img, bg: 255 - cv2.bitwise_and(gray, gray, mask=bg),
    "_inv_masked_bin":      lambda gray, bin_img, bg: 255 - cv2.bitwise_and(bin_img, bin_img, mask=bg),
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DATASET_PAIRS = (
    ("ICDAR2017_CLaMM_Training", "ICDAR2017_CLaMM_Training_augmented"),
    ("ICDAR2017_CLaMM_evaluation_set", "ICDAR2017_CLaMM_evaluation_set_augmented"),
)

# ──-------------------------------------------------------------------------
# 🚀  Helpers
# ────────────────────────────────────────────────────────────────────────────


def process_page(img_path: Path, dataset_dir: Path, aug_dataset_dir: Path):
    """
    Mirrors the folder structure of dataset_dir inside aug_dataset_dir.
    Saves:
        - the original image (saved as JPG)
        - all augmented variants
    """

    # ---------------------------------------------------------
    # 1. Build output directory
    # ---------------------------------------------------------
    out_dir = aug_dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Load & preprocess
    # ---------------------------------------------------------
    gray, bin_img, inv_bin = load_and_preprocess_image(str(img_path))

    # ---------------------------------------------------------
    # 3. Save the original image as JPG in the aug dataset directory
    # ---------------------------------------------------------
    dst_original = out_dir / f"{img_path.stem}.jpg"
    cv2.imwrite(str(dst_original), gray)

    ths_low  = bin_img.shape[0] * 0.001
    ths_high = bin_img.shape[0] * 0.1
    height_range = estimate_binary_height(inv_bin, ths_low, ths_high)
    if height_range is None:
        raise ValueError("Could not estimate binary height from connected components")

    avr_h = max(MIN_TEXT_HEIGHT, height_range[0])
    eta      = ETA_FACTOR * avr_h
    bg_mask  = (filter_regions(bin_img, SCALE, eta, THETAS) < 0.99).astype(np.uint8) * 255

    # ---------------------------------------------------------
    # 4. Generate & save each variant
    # ---------------------------------------------------------
    for suffix, fn in VARIANTS.items():

        variant_img = fn(gray, bin_img, bg_mask)

        out_name = f"{img_path.stem}{suffix}.jpg"
        cv2.imwrite(str(out_dir / out_name), variant_img)


# ──-------------------------------------------------------------------------
# 🏃‍♀️  Main driver
# ────────────────────────────────────────────────────────────────────────────

def augment_ds(ds_dir: Path, aug_ds_dir: Path):
    if not ds_dir.exists():
        print(f"⏭  {ds_dir} not found – skipping.")
        return
    print(f"Source: {ds_dir}")
    print(f"Target: {aug_ds_dir}")
    pages = [p for p in ds_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"▶   {len(pages):6d} pages")

    for p in pages:
        try:
            process_page(p, ds_dir, aug_ds_dir)
        except Exception as e:
            print(f"[WARN] {p}: {e}")

def main():
    for ds_name, aug_name in DATASET_PAIRS:
        augment_ds(Path(ds_name), Path(aug_name))
        print()


if __name__ == "__main__":
    main()
