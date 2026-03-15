from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lib.common import ensure_dir, load_yaml


METRICS = ["dice", "hd95", "lesion_recall", "lesion_precision", "lesion_f1", "volume_relative_error"]


def summarize_subset(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_cases": 0, "n_patients": 0}
    row = {
        "n_cases": int(df.shape[0]),
        "n_patients": int(df["patient_id"].nunique()),
    }
    for metric in METRICS:
        if metric in df.columns:
            row[f"{metric}_mean"] = float(df[metric].mean())
            row[f"{metric}_std"] = float(df[metric].std(ddof=0))
    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize ROI-constrained single-sequence screening experiments.")
    parser.add_argument("--study-config", default="configs/roi_study_config.yaml")
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--require-external-for-anchor", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    experiments_root = Path(args.experiments_root)
    out_dir = ensure_dir(args.out_dir)
    rows = []

    for experiment_id, exp in cfg.get("experiments", {}).items():
        if exp.get("stage") != "screening":
            continue
        metrics_path = experiments_root / experiment_id / "results" / "per_case_metrics.csv"
        if not metrics_path.exists():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "primary_target": exp.get("primary_target", ""),
                    "source_tag": exp.get("source_tag", experiment_id),
                    "status": "missing_metrics",
                }
            )
            continue

        metrics = pd.read_csv(metrics_path)
        metrics = metrics[metrics.get("missing_prediction", 0) == 0].copy()
        primary_target = exp.get("primary_target", "")
        metrics = metrics[metrics["target_seq"].astype(str) == str(primary_target)].copy()

        internal = metrics[metrics["eval_mode"] == "internal_cv"].copy()
        external = metrics[metrics["eval_mode"] == "external_test"].copy()
        internal_summary = summarize_subset(internal)
        external_summary = summarize_subset(external)

        row = {
            "experiment_id": experiment_id,
            "primary_target": primary_target,
            "source_tag": exp.get("source_tag", experiment_id),
            "dataset_id": int(exp["dataset_id"]),
            "status": "ok",
        }
        row.update({f"internal_{k}": v for k, v in internal_summary.items()})
        row.update({f"external_{k}": v for k, v in external_summary.items()})
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No screening experiments were found in the config.")

    required_numeric_cols = [
        "external_dice_mean",
        "internal_dice_mean",
        "external_hd95_mean",
        "external_n_cases",
    ]
    for col in required_numeric_cols:
        if col not in summary.columns:
            summary[col] = np.nan

    summary["anchor_sort_external_dice"] = summary["external_dice_mean"].fillna(-1.0)
    summary["anchor_sort_internal_dice"] = summary["internal_dice_mean"].fillna(-1.0)
    summary["anchor_sort_external_hd95"] = summary["external_hd95_mean"].fillna(np.inf)

    eligible = summary[summary["status"] == "ok"].copy()
    if args.require_external_for_anchor:
        eligible = eligible[eligible["external_n_cases"].fillna(0) > 0].copy()
    if eligible.empty:
        raise ValueError("No eligible experiments found for anchor selection.")

    eligible = eligible.sort_values(
        by=["anchor_sort_external_dice", "anchor_sort_internal_dice", "anchor_sort_external_hd95"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    eligible["rank"] = np.arange(1, len(eligible) + 1)

    summary = summary.merge(eligible[["experiment_id", "rank"]], on="experiment_id", how="left")
    summary = summary.sort_values(by=["rank", "primary_target"], na_position="last").reset_index(drop=True)

    summary_path = out_dir / "sequence_screening_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    recommended = eligible.iloc[0].to_dict()
    recommended_anchor = {
        "recommended_anchor_experiment_id": recommended["experiment_id"],
        "recommended_anchor_target": recommended["primary_target"],
        "recommended_source_tag": recommended["source_tag"],
        "ranking_metric": cfg.get("post_registration", {}).get("expected_anchor_metric", "external_dice_mean"),
    }
    with open(out_dir / "recommended_anchor.json", "w", encoding="utf-8") as handle:
        json.dump(recommended_anchor, handle, indent=2, ensure_ascii=False)

    print(f"Saved screening summary to: {summary_path}")
    print(f"Recommended anchor target: {recommended_anchor['recommended_anchor_target']}")


if __name__ == "__main__":
    main()

