from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from aggregated tables.")
    parser.add_argument("--paper-dir", required=True)
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)

    # Fig 3 / Fig 4 based on Table 3
    table3_path = paper_dir / "Table3_cross_sequence.csv"
    if table3_path.exists():
        table3 = pd.read_csv(table3_path)

        # Figure 4: cross-sequence heatmap (Dice)
        heat = table3.pivot(index="source_seq", columns="target_seq", values="dice_mean")
        plt.figure(figsize=(10, 4))
        plt.imshow(heat.values, aspect="auto")
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.colorbar(label="Dice")
        plt.tight_layout()
        plt.savefig(paper_dir / "Figure4_cross_sequence_heatmap.png", dpi=200)
        plt.close()

        # Figure 5-like line plot (Dice by target)
        plt.figure(figsize=(10, 4))
        for src, sub in table3.groupby("source_seq"):
            plt.plot(sub["target_seq"], sub["dice_mean"], marker="o", label=src)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Dice")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper_dir / "Figure3_source_target_dice.png", dpi=200)
        plt.close()

    table2_path = paper_dir / "Table2_main_source_cv.csv"
    if table2_path.exists():
        table2 = pd.read_csv(table2_path)
        if "dice_mean" in table2.columns:
            plt.figure(figsize=(5, 4))
            plt.bar(table2["source_seq"], table2["dice_mean"])
            plt.ylabel("Dice")
            plt.tight_layout()
            plt.savefig(paper_dir / "Figure2_main_source_bar.png", dpi=200)
            plt.close()

    print(f"Saved figures to: {paper_dir}")


if __name__ == "__main__":
    main()
