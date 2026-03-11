from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import KFold


def main():
    parser = argparse.ArgumentParser(description="Assign strict patient-level folds.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    patients = sorted(df["patient_id"].astype(str).unique().tolist())

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    patient_to_fold = {}
    for fold_id, (_, test_idx) in enumerate(kf.split(patients)):
        for i in test_idx:
            patient_to_fold[patients[i]] = fold_id

    df["fold"] = df["patient_id"].astype(str).map(patient_to_fold)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    fold_table = (
        df.groupby(["fold"])["patient_id"]
        .nunique()
        .reset_index()
        .rename(columns={"patient_id": "n_patients"})
    )
    print(fold_table.to_string(index=False))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
