from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VARIANT_LABELS = {
    "": "Original",
    "_inverse": "Inverted grayscale",
    "_masked_inv_gray": "Masked inverted gray",
    "_inv_masked_gray": "Inverted masked gray",
    "_inv_masked_bin": "Inverted masked binary",
}


def collect_images(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def choose_evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    if count <= 1:
        return [paths[len(paths) // 2]]

    step = (len(paths) - 1) / (count - 1)
    indices = [round(i * step) for i in range(count)]
    return [paths[index] for index in indices]


def load_label_groups(labels_csv: Path) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    with labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            filename = row.get("FILENAME") or row.get("FILE_NAME")
            label = row.get("SCRIPT_TYPE") or row.get("Script_type_ICDAR2017")
            if not filename or not label:
                continue
            groups[int(label)].append(Path(filename).name)
    return dict(sorted(groups.items()))


def choose_per_label_samples(dataset_dir: Path, labels_csv: Path) -> list[tuple[int, Path]]:
    grouped_filenames = load_label_groups(labels_csv)
    selected: list[tuple[int, Path]] = []
    for label, filenames in grouped_filenames.items():
        chosen_name = filenames[len(filenames) // 2]
        image_path = dataset_dir / chosen_name
        if image_path.exists():
            selected.append((label, image_path))
    return selected


def choose_page_with_all_variants(dataset_dir: Path) -> str:
    grouped: dict[str, set[str]] = defaultdict(set)
    for path in collect_images(dataset_dir):
        stem = path.stem
        matched_suffix = None
        for suffix in VARIANT_LABELS:
            if suffix and stem.endswith(suffix):
                grouped[stem[: -len(suffix)]].add(suffix)
                matched_suffix = suffix
                break
        if matched_suffix is None:
            grouped[stem].add("")

    for page_stem in sorted(grouped):
        if set(VARIANT_LABELS) <= grouped[page_stem]:
            return page_stem
    raise FileNotFoundError("No page with the full augmentation set was found.")


def build_augmentation_entries(dataset_dir: Path, page_stem: str) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for suffix, label in VARIANT_LABELS.items():
        image_path = dataset_dir / f"{page_stem}{suffix}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing augmentation variant: {image_path}")
        entries.append((label, image_path))
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a publication-style figure with unaugmented dataset samples."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("ICDAR2017_CLaMM_Training"),
        help="Directory containing the original, non-augmented dataset images.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv"),
        help="CSV file containing the numeric script labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report_figures/script_label_comparison.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--mode",
        choices=("per-label", "gallery", "augmentations"),
        default="per-label",
        help="Render one sample per numeric label or a general dataset gallery.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=6,
        help="Number of images to display when mode=gallery.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns in the figure grid.",
    )
    parser.add_argument(
        "--show-filenames",
        action="store_true",
        help="Show filenames under each panel.",
    )
    parser.add_argument(
        "--page-stem",
        type=str,
        default=None,
        help="Page stem to use when mode=augmentations. If omitted, a valid page is chosen automatically.",
    )
    return parser.parse_args()


def style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#b8b8b8")


def crop_to_content(image: Image.Image, margin_ratio: float = 0.03) -> Image.Image:
    grayscale = image.convert("L")
    arr = np.asarray(grayscale)
    mask = arr < 235
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0)
    height, width = arr.shape

    margin_y = max(12, int(height * margin_ratio))
    margin_x = max(12, int(width * margin_ratio))

    top = max(0, top - margin_y)
    left = max(0, left - margin_x)
    bottom = min(height - 1, bottom + margin_y)
    right = min(width - 1, right + margin_x)

    return image.crop((left, top, right + 1, bottom + 1))


def build_figure(entries: list[tuple[str, Path]], columns: int, output: Path, show_filenames: bool, title: str) -> None:
    rows = ceil(len(entries) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 5.2 * rows))
    fig.patch.set_facecolor("white")

    if not hasattr(axes, "flat"):
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, (panel_title, image_path) in zip(axes, entries):
        with Image.open(image_path) as image:
            cropped = crop_to_content(image)
            ax.imshow(cropped, cmap="gray")
        style_axis(ax)
        ax.set_title(panel_title, fontsize=11, fontweight="semibold", pad=8)
        if show_filenames:
            ax.set_xlabel(image_path.stem, fontsize=8, color="#444444", labelpad=8)

    for ax in axes[len(entries):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="semibold", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, wspace=0.06, hspace=0.14)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir

    if args.mode == "augmentations" and dataset_dir == Path("ICDAR2017_CLaMM_Training"):
        dataset_dir = Path("ICDAR2017_CLaMM_Training_augmented")

    if args.mode == "per-label":
        selected = choose_per_label_samples(dataset_dir, args.labels_csv)
        if not selected:
            raise FileNotFoundError("No label-linked images were found.")
        entries = [(f"Script label {label}", path) for label, path in selected]
        title = "Representative Unaugmented Samples by Numeric Script Label"
    elif args.mode == "augmentations":
        page_stem = args.page_stem or choose_page_with_all_variants(dataset_dir)
        entries = build_augmentation_entries(dataset_dir, page_stem)
        title = f"Augmentation Variants for Page {page_stem}"
    else:
        image_paths = collect_images(dataset_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {dataset_dir}")
        selected_paths = choose_evenly_spaced(image_paths, args.num_images)
        entries = [(f"Sample {idx + 1}", path) for idx, path in enumerate(selected_paths)]
        title = "Sample Unaugmented Images from the ICDAR2017 CLaMM Training Set"

    build_figure(
        entries=entries,
        columns=max(1, args.columns),
        output=args.output,
        show_filenames=args.show_filenames,
        title=title,
    )

    print(f"Saved figure to {args.output}")
    print("Selected panels:")
    for panel_title, path in entries:
        print(f" - {panel_title}: {path.name}")


if __name__ == "__main__":
    main()
