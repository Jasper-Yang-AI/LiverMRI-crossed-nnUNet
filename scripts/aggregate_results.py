from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ensure_dir


METRICS = ["dice", "hd95", "lesion_recall", "lesion_precision", "lesion_f1", "volume_relative_error"]


def agg_metric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out_rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_cases"] = int(sub.shape[0])
        row["n_patients"] = int(sub["patient_id"].nunique()) if "patient_id" in sub.columns else int(sub.shape[0])
        for metric in METRICS:
            if metric in sub.columns:
                row[f"{metric}_mean"] = float(sub[metric].mean())
                row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def save_table(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics into paper-ready tables.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--audit", required=False, default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    metrics = pd.read_csv(args.metrics)
    valid = metrics[metrics.get("missing_prediction", 0) == 0].copy()
    if valid.empty:
        raise ValueError("No valid predictions found in metrics table.")

    if "eval_mode" not in valid.columns:
        valid["eval_mode"] = "legacy"

    internal = valid[valid["eval_mode"] == "internal_cv"].copy()
    external = valid[valid["eval_mode"] == "external_test"].copy()

    if "source_primary_target" in internal.columns:
        same_domain = internal[
            internal["source_primary_target"].fillna("").astype(str).str.len() > 0
        ].copy()
        same_domain = same_domain[same_domain["source_primary_target"] == same_domain["target_seq"]].copy()
        cross_domain = internal.copy()
        has_primary = cross_domain["source_primary_target"].fillna("").astype(str).str.len() > 0
        cross_domain = cross_domain[(~has_primary) | (cross_domain["source_primary_target"] != cross_domain["target_seq"])].copy()
    else:
        same_domain = internal[internal["source_seq"] == internal["target_seq"]].copy()
        cross_domain = internal[internal["source_seq"] != internal["target_seq"]].copy()

    table2 = agg_metric(same_domain, ["source_seq", "target_seq"])
    table3 = agg_metric(cross_domain, ["source_seq", "target_seq"])
    table4 = agg_metric(external, ["source_seq", "target_seq"])
    combined = agg_metric(valid, ["eval_mode", "source_seq", "target_seq"])
    raw_breakdown = (
        agg_metric(valid, ["eval_mode", "source_seq", "target_seq", "raw_seq_group"])
        if "raw_seq_group" in valid.columns
        else pd.DataFrame()
    )

    macro_rows = []
    for (eval_mode, source_seq), sub in combined.groupby(["eval_mode", "source_seq"]):
        row = {"eval_mode": eval_mode, "source_seq": source_seq}
        for metric in [
            "dice_mean",
            "hd95_mean",
            "lesion_recall_mean",
            "lesion_precision_mean",
            "lesion_f1_mean",
            "volume_relative_error_mean",
        ]:
            if metric in sub.columns:
                row[f"macro_{metric}"] = float(sub[metric].mean())
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    save_table(table2, out_dir / "Table2_internal_source_cv.csv")
    save_table(table3, out_dir / "Table3_internal_cross_sequence.csv")
    save_table(table4, out_dir / "Table4_external_test.csv")
    save_table(combined, out_dir / "Table_all_eval_modes.csv")
    save_table(macro, out_dir / "Table_macro_average.csv")
    save_table(raw_breakdown, out_dir / "Table_bundle_raw_breakdown.csv")

    if args.audit:
        audit_path = Path(args.audit)
        if audit_path.exists():
            with open(audit_path, "r", encoding="utf-8") as handle:
                audit = json.load(handle)
            with open(out_dir / "AuditSummary.json", "w", encoding="utf-8") as handle:
                json.dump(audit, handle, indent=2, ensure_ascii=False)

    summary = {
        "n_total_evals": int(valid.shape[0]),
        "n_internal_cv_evals": int(internal.shape[0]),
        "n_external_test_evals": int(external.shape[0]),
        "sources": sorted(valid["source_seq"].dropna().unique().tolist()),
        "targets": sorted(valid["target_seq"].dropna().unique().tolist()),
        "eval_modes": sorted(valid["eval_mode"].dropna().unique().tolist()),
    }
    with open(out_dir / "study_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved paper tables to: {out_dir}")


if __name__ == "__main__":
    main()
