from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Select best / median / worst cases for qualitative figure assembly.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    df = df[df.get("missing_prediction", 0) == 0].copy()
    rows = []

    for (source_seq, target_seq), sub in df.groupby(["source_seq", "target_seq"]):
        sub = sub.sort_values("dice")
        worst = sub.head(args.topk)
        best = sub.tail(args.topk)
        mid = sub.iloc[max(len(sub)//2 - args.topk//2, 0): max(len(sub)//2 - args.topk//2, 0) + args.topk]
        for tag, block in [("worst", worst), ("median", mid), ("best", best)]:
            tmp = block.copy()
            tmp["rank_group"] = tag
            rows.append(tmp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved failure-case manifest: {out}")


if __name__ == "__main__":
    main()
