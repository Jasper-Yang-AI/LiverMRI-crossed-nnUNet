from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import ensure_dir, load_yaml


def main():
    parser = argparse.ArgumentParser(description="Prepare few-shot target-domain split manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--target", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--subset-values", nargs="+", default=["train", "imagesTr"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if "subset" not in df.columns:
        df["subset"] = "train"
    sub = df[(df["seq_group"] == args.target) & (df["subset"].astype(str).isin(args.subset_values))].copy()
    patients = sorted(sub["patient_id"].astype(str).unique().tolist())
    if not patients:
        raise ValueError(f"No cases found for target {args.target} in subsets {args.subset_values}")

    train_p, holdout_p = train_test_split(patients, train_size=args.fraction, random_state=args.seed)
    sub["fewshot_role"] = sub["patient_id"].astype(str).apply(lambda x: "adapt" if x in train_p else "holdout")

    out_dir = ensure_dir(args.out_dir)
    out_csv = out_dir / f"fewshot_{args.target}_{str(args.fraction).replace('.', 'p')}.csv"
    sub.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved few-shot split manifest: {out_csv}")


if __name__ == "__main__":
    main()
