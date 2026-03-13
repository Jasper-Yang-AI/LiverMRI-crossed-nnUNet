from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_heatmap(table: pd.DataFrame, out_path: Path, title: str) -> None:
    if table.empty:
        return
    heat = table.pivot(index="source_seq", columns="target_seq", values="dice_mean")
    plt.figure(figsize=(10, 4))
    plt.imshow(heat.values, aspect="auto")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.title(title)
    plt.colorbar(label="Dice")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_source_bar(table: pd.DataFrame, out_path: Path, title: str) -> None:
    if table.empty or "dice_mean" not in table.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(table["source_seq"], table["dice_mean"])
    plt.ylabel("Dice")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_target_bar(table: pd.DataFrame, out_path: Path, title: str) -> None:
    if table.empty or "dice_mean" not in table.columns:
        return
    plt.figure(figsize=(10, 4))
    for source_seq, sub in table.groupby("source_seq"):
        plt.plot(sub["target_seq"], sub["dice_mean"], marker="o", label=source_seq)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Dice")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from aggregated tables.")
    parser.add_argument("--paper-dir", required=True)
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)

    table2_path = paper_dir / "Table2_internal_source_cv.csv"
    table3_path = paper_dir / "Table3_internal_cross_sequence.csv"
    table4_path = paper_dir / "Table4_external_test.csv"

    if table2_path.exists():
        plot_source_bar(pd.read_csv(table2_path), paper_dir / "Figure2_internal_source_cv.png", "Internal same-sequence CV")

    if table3_path.exists():
        table3 = pd.read_csv(table3_path)
        plot_heatmap(table3, paper_dir / "Figure3_internal_cross_sequence_heatmap.png", "Internal cross-sequence transfer")
        plot_target_bar(table3, paper_dir / "Figure3_internal_cross_sequence_lines.png", "Internal cross-sequence Dice")

    if table4_path.exists():
        table4 = pd.read_csv(table4_path)
        plot_heatmap(table4, paper_dir / "Figure4_external_test_heatmap.png", "External zero-shot transfer")
        plot_target_bar(table4, paper_dir / "Figure4_external_test_lines.png", "External zero-shot Dice")

    print(f"Saved figures to: {paper_dir}")


if __name__ == "__main__":
    main()
