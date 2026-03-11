from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create nnUNetv2 splits_final.json with strict patient-level fold assignment.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seq-group", required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--nnunet-preprocessed", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["seq_group"] == args.seq_group].copy()
    if "fold" not in df.columns:
        raise ValueError("Manifest must contain `fold` column.")

    def make_case_id(row):
        safe_patient = str(row["patient_id"]).replace(" ", "_")
        safe_seq = str(row["seq_group"]).replace("+", "PLUS").replace("-", "")
        return f"{safe_patient}__{safe_seq}__{int(row.name):05d}"

    df["case_id"] = df.apply(make_case_id, axis=1)

    splits = []
    for fold in sorted(df["fold"].unique().tolist()):
        val_cases = df[df["fold"] == fold]["case_id"].tolist()
        train_cases = df[df["fold"] != fold]["case_id"].tolist()
        splits.append({"train": train_cases, "val": val_cases})

    dataset_dir = Path(args.nnunet_preprocessed) / f"Dataset{args.dataset_id:03d}_LiverTumor_{args.seq_group}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / "splits_final.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"Saved splits_final.json to {dataset_dir}")


if __name__ == "__main__":
    main()
