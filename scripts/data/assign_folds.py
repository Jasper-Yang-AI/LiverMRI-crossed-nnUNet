from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import KFold


def main():
    parser = argparse.ArgumentParser(description="Assign strict patient-level folds.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--train-subset-values",
        nargs="+",
        default=["train", "imagesTr"],
        help="Only patients in these subset values receive CV folds. Other rows are kept as external test with fold=-1.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if "subset" not in df.columns:
        df["subset"] = "train"

    train_mask = df["subset"].astype(str).isin(args.train_subset_values)
    train_patients = sorted(df.loc[train_mask, "patient_id"].astype(str).unique().tolist())
    if not train_patients:
        raise ValueError("No training patients found. Check `subset` values or pass `--train-subset-values`.")

    if len(train_patients) < args.n_folds:
        raise ValueError(f"Need at least {args.n_folds} training patients, but only found {len(train_patients)}.")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    patient_to_fold = {}
    for fold_id, (_, test_idx) in enumerate(kf.split(train_patients)):
        for idx in test_idx:
            patient_to_fold[train_patients[idx]] = fold_id

    df["fold"] = -1
    df.loc[train_mask, "fold"] = df.loc[train_mask, "patient_id"].astype(str).map(patient_to_fold).astype(int)
    df["cohort_role"] = df["subset"].astype(str).apply(
        lambda subset: "internal_cv" if subset in args.train_subset_values else "external_test"
    )

    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    fold_table = (
        df[train_mask]
        .groupby(["fold"])["patient_id"]
        .nunique()
        .reset_index()
        .rename(columns={"patient_id": "n_patients"})
    )
    subset_table = (
        df.groupby(["subset", "cohort_role"])["patient_id"]
        .nunique()
        .reset_index()
        .rename(columns={"patient_id": "n_patients"})
    )
    print("Training fold counts")
    print(fold_table.to_string(index=False))
    print("")
    print("Subset summary")
    print(subset_table.to_string(index=False))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

