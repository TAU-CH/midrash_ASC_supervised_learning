
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

DS_DIR = Path("ASC_dataset_v3_augmented")
PATCHIFIED_DS_DIR = Path("ASC_dataset_v3_augmented_patchified")
PATCH_SIZE = 224
STRIDE = 224     # or e.g. 112 for overlapping patches
BG_THRESHOLD = 0.98  # drop patches with ≥98% near-white (tune as needed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_mostly_background(patch: Image.Image) -> bool:
    arr = np.array(patch.convert("L")) / 255.0
    # fraction of pixels brighter than 0.95
    frac_white = (arr > 0.95).mean()
    return frac_white > BG_THRESHOLD

def patchify_dir(src_dir: Path, dst_dir: Path):
    # Assuming structure: src_dir/class_name/*.png
    split = src_dir.name  # e.g. "train" / "val" / "test"
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name  # e.g. "french" / "german"
        out_dir = dst_dir / label
        ensure_dir(out_dir)

        for img_path in tqdm(list(class_dir.glob("*.*")), desc=f"{split} / {label}"):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            base_name = img_path.stem

            patch_idx = 0
            for top in range(0, h - PATCH_SIZE + 1, STRIDE):
                for left in range(0, w - PATCH_SIZE + 1, STRIDE):
                    patch = img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
                    if is_mostly_background(patch):
                        continue
                    patch_name = f"{base_name}_y{top:05d}_x{left:05d}_{patch_idx:04d}.png"
                    patch.save(out_dir / patch_name)
                    patch_idx += 1
for split in ["train", "val", "test"]:
    patchify_dir(DS_DIR / split, PATCHIFIED_DS_DIR / split)
