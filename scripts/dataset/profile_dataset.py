from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.common.common import ensure_dir, expand_named_collection, load_yaml, resolve_experiment


def summarize_patient_signatures(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, patient_id), sub in df.groupby(["subset", "patient_id"]):
        signature = "+".join(sorted(sub["seq_group"].fillna("UNKNOWN").unique().tolist()))
        rows.append(
            {
                "subset": subset,
                "patient_id": patient_id,
                "n_sequences": int(sub["seq_group"].nunique()),
                "sequence_signature": signature,
            }
        )
    return pd.DataFrame(rows)


def summarize_bundle_counts(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for bundle_name in cfg.get("target_bundles", {}):
        members = expand_named_collection(bundle_name, cfg, sections=("target_bundles", "sequence_bundles"))
        sub = df[df["seq_group"].isin(members)].copy()
        for subset, block in sub.groupby("subset"):
            rows.append(
                {
                    "bundle": bundle_name,
                    "subset": subset,
                    "members": "+".join(members),
                    "n_cases": int(block.shape[0]),
                    "n_patients": int(block["patient_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def summarize_experiment_catalog(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    train_df = df[df["cohort_role"] == "internal_cv"].copy()
    test_df = df[df["cohort_role"] == "external_test"].copy()

    for experiment_id in cfg.get("experiments", {}):
        exp = resolve_experiment(cfg, experiment_id)
        source_members = expand_named_collection(exp["source_bundle"], cfg, sections=("sequence_bundles", "target_bundles"))
        train_source = train_df[train_df["seq_group"].isin(source_members)].copy()
        rows.append(
            {
                "experiment_id": experiment_id,
                "description": exp.get("description", ""),
                "dataset_id": int(exp["dataset_id"]),
                "source_tag": exp.get("source_tag", experiment_id),
                "source_bundle": exp["source_bundle"],
                "source_targets": "+".join(exp.get("source_targets", [])),
                "source_members": "+".join(source_members),
                "n_train_cases": int(train_source.shape[0]),
                "n_train_patients": int(train_source["patient_id"].nunique()),
                "targets": "+".join(exp.get("targets", [])),
                "n_external_patients_total": int(test_df["patient_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Profile manifest composition for study planning.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/dataset/livermri_crossseq_dataset.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    df = pd.read_csv(args.manifest)
    if "subset" not in df.columns:
        df["subset"] = "train"
    if "cohort_role" not in df.columns:
        df["cohort_role"] = df["subset"].astype(str).apply(lambda subset: "internal_cv" if subset in {"train", "imagesTr"} else "external_test")

    out_dir = ensure_dir(args.out_dir)

    subset_sequence = (
        df.groupby(["subset", "seq_group"])["patient_id"]
        .nunique()
        .reset_index()
        .rename(columns={"patient_id": "n_patients"})
        .sort_values(["subset", "n_patients", "seq_group"], ascending=[True, False, True])
    )
    subset_sequence.to_csv(out_dir / "subset_sequence_counts.csv", index=False, encoding="utf-8-sig")

    bundle_counts = summarize_bundle_counts(df, cfg)
    bundle_counts.to_csv(out_dir / "target_bundle_counts.csv", index=False, encoding="utf-8-sig")

    signature_df = summarize_patient_signatures(df)
    signature_summary = (
        signature_df.groupby(["subset", "n_sequences", "sequence_signature"])
        .size()
        .reset_index(name="n_patients")
        .sort_values(["subset", "n_patients"], ascending=[True, False])
    )
    signature_summary.to_csv(out_dir / "patient_sequence_signatures.csv", index=False, encoding="utf-8-sig")

    train_df = df[df["cohort_role"] == "internal_cv"].copy()
    source_candidates = (
        train_df.groupby("seq_group")
        .agg(n_cases=("patient_id", "size"), n_patients=("patient_id", "nunique"))
        .reset_index()
        .sort_values(["n_patients", "n_cases", "seq_group"], ascending=[False, False, True])
    )
    source_candidates["is_default_source"] = source_candidates["seq_group"] == cfg.get("default_source_sequence", "")
    source_candidates.to_csv(out_dir / "source_candidate_ranking.csv", index=False, encoding="utf-8-sig")

    experiment_catalog = summarize_experiment_catalog(df, cfg)
    experiment_catalog.to_csv(out_dir / "experiment_catalog.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_rows": int(df.shape[0]),
        "n_patients": int(df["patient_id"].nunique()),
        "subset_counts": df["subset"].value_counts().to_dict(),
        "cohort_counts": df["cohort_role"].value_counts().to_dict(),
        "default_source_sequence": cfg.get("default_source_sequence"),
        "default_experiment": cfg.get("default_experiment"),
        "top_source_candidates": source_candidates.head(5).to_dict(orient="records"),
    }
    (out_dir / "dataset_profile_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Dataset Profile",
        "",
        f"- Total rows: {summary['n_rows']}",
        f"- Total patients: {summary['n_patients']}",
        f"- Default source sequence in config: {summary['default_source_sequence']}",
        f"- Default experiment in config: {summary['default_experiment']}",
        "",
        "## Subset counts",
    ]
    for subset, count in summary["subset_counts"].items():
        lines.append(f"- {subset}: {count}")
    lines.extend(["", "## Group-level target counts"])
    for _, row in bundle_counts.iterrows():
        lines.append(
            f"- {row['bundle']} / {row['subset']}: {int(row['n_patients'])} patients / {int(row['n_cases'])} cases ({row['members']})"
        )

    lines.extend(["", "## Top raw-sequence source candidates"])
    for _, row in source_candidates.head(5).iterrows():
        tag = " (default)" if bool(row["is_default_source"]) else ""
        lines.append(f"- {row['seq_group']}: {int(row['n_patients'])} patients / {int(row['n_cases'])} cases{tag}")

    lines.extend(["", "## Experiment catalog"])
    for _, row in experiment_catalog.iterrows():
        lines.append(
            f"- {row['experiment_id']} (dataset {int(row['dataset_id'])}): {row['description']} | train {int(row['n_train_patients'])} patients / {int(row['n_train_cases'])} cases | logical source {row['source_targets']} | raw members {row['source_members']}"
        )
    (out_dir / "DATASET_PROFILE.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved dataset profile to: {out_dir}")


if __name__ == "__main__":
    main()

