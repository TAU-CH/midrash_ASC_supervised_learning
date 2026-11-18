from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = (
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
)


def _history_to_frame(history: Mapping[str, Sequence[float]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        df = history.copy()
    else:
        df = pd.DataFrame(history)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Training history is missing columns: {missing}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    df["epoch"] = df["epoch"].astype(int)
    return df.sort_values("epoch").reset_index(drop=True)


def plot_training_history(
    history: Mapping[str, Sequence[float]] | pd.DataFrame,
    output_dir: str | Path = "report_figures",
    stem: str = "mil_training_curves",
    title: str = "MIL CNN Training Dynamics",
) -> tuple[Path, Path, Path]:
    """
    Save publication-style training curves as PNG, PDF, and CSV.
    """
    df = _history_to_frame(history)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    df.to_csv(csv_path, index=False)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    fig.patch.set_facecolor("white")
    colors = {
        "train": "#1f77b4",
        "val": "#d62728",
        "accent": "#2f2f2f",
    }

    metric_specs = [
        ("loss", "Cross-Entropy Loss"),
        ("acc", "Accuracy"),
    ]

    for ax, (suffix, ylabel) in zip(axes, metric_specs):
        train_col = f"train_{suffix}"
        val_col = f"val_{suffix}"

        ax.plot(
            df["epoch"],
            df[train_col],
            color=colors["train"],
            linewidth=2.2,
            marker="o",
            markersize=4.8,
            label="Train",
        )
        ax.plot(
            df["epoch"],
            df[val_col],
            color=colors["val"],
            linewidth=2.2,
            marker="s",
            markersize=4.4,
            label="Validation",
        )

        best_idx = df[val_col].idxmax() if suffix == "acc" else df[val_col].idxmin()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_value = float(df.loc[best_idx, val_col])

        ax.scatter(
            [best_epoch],
            [best_value],
            s=70,
            color=colors["accent"],
            zorder=5,
            label="Best validation" if suffix == "loss" else None,
        )
        ax.annotate(
            f"best @ {best_epoch}\n{best_value:.3f}",
            xy=(best_epoch, best_value),
            xytext=(8, -10 if suffix == "loss" else 12),
            textcoords="offset points",
            fontsize=9,
            color=colors["accent"],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.95,
            },
        )

        ax.set_title(f"{ylabel} vs. Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(df["epoch"])
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) >= 2:
        fig.legend(
            handles[:2],
            labels[:2],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.91),
            borderaxespad=0.0,
        )

    fig.suptitle(title, fontsize=15, fontweight="semibold", y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.145, top=0.765, wspace=0.24)
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return png_path, pdf_path, csv_path


def load_history_json(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot publication-style train/validation curves.")
    parser.add_argument("history_json", type=Path, help="JSON file with epoch/train_loss/train_acc/val_loss/val_acc")
    parser.add_argument("--output-dir", type=Path, default=Path("report_figures"))
    parser.add_argument("--stem", type=str, default="mil_training_curves")
    parser.add_argument("--title", type=str, default="MIL CNN Training Dynamics")
    args = parser.parse_args()

    history = load_history_json(args.history_json)
    png_path, pdf_path, csv_path = plot_training_history(
        history=history,
        output_dir=args.output_dir,
        stem=args.stem,
        title=args.title,
    )
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved CSV: {csv_path}")
