from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ensure_dir


def agg_metric(df: pd.DataFrame, group_cols):
    metrics = ["dice", "hd95", "lesion_recall", "lesion_precision", "lesion_f1", "volume_relative_error"]
    out_rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_cases"] = int(sub.shape[0])
        for m in metrics:
            if m in sub.columns:
                row[f"{m}_mean"] = float(sub[m].mean())
                row[f"{m}_std"] = float(sub[m].std(ddof=0))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics into paper-ready tables.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--audit", required=False, default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    metrics = pd.read_csv(args.metrics)

    valid = metrics[metrics.get("missing_prediction", 0) == 0].copy()

    # Table 2: source same-sequence if present
    source_same = valid[valid["source_seq"] == valid["target_seq"]].copy()
    table2 = agg_metric(source_same, ["source_seq"])

    # Table 3: cross-sequence
    table3 = agg_metric(valid, ["source_seq", "target_seq"])

    # Macro average over targets for each source
    macro_rows = []
    for source_seq, sub in table3.groupby("source_seq"):
        row = {"source_seq": source_seq}
        for metric in ["dice_mean", "hd95_mean", "lesion_recall_mean", "lesion_precision_mean", "lesion_f1_mean"]:
            if metric in sub.columns:
                row[f"macro_{metric}"] = float(sub[metric].mean())
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    table2.to_csv(out_dir / "Table2_main_source_cv.csv", index=False, encoding="utf-8-sig")
    table3.to_csv(out_dir / "Table3_cross_sequence.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(out_dir / "Table3_macro_average.csv", index=False, encoding="utf-8-sig")

    if args.audit:
        audit_path = Path(args.audit)
        if audit_path.exists():
            with open(audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
            with open(out_dir / "AuditSummary.json", "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2, ensure_ascii=False)

    # Study summary for manuscript templating
    summary = {
        "n_total_evals": int(valid.shape[0]),
        "sources": sorted(valid["source_seq"].dropna().unique().tolist()),
        "targets": sorted(valid["target_seq"].dropna().unique().tolist()),
    }
    with open(out_dir / "study_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved paper tables to: {out_dir}")


if __name__ == "__main__":
    main()
