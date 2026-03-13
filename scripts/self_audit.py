from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ensure_dir, load_yaml


def main():
    parser = argparse.ArgumentParser(description="Objective self-audit for publication readiness.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    out_dir = ensure_dir(args.out_dir)
    df = pd.read_csv(args.manifest)
    if "subset" not in df.columns:
        df["subset"] = "train"
    if "cohort_role" not in df.columns:
        df["cohort_role"] = "internal_cv"

    findings = []
    status = "pass"

    # leakage
    if "fold" in df.columns:
        internal = df[df["cohort_role"] == "internal_cv"].copy()
        p_fold_counts = internal.groupby("patient_id")["fold"].nunique() if not internal.empty else pd.Series(dtype=int)
        leaked = p_fold_counts[p_fold_counts > 1]
        if len(leaked) > 0:
            findings.append({"level": "error", "item": "patient_leakage", "count": int(len(leaked))})
            status = "fail"
        else:
            findings.append({"level": "ok", "item": "patient_leakage", "count": 0})
    else:
        findings.append({"level": "warning", "item": "missing_fold_column"})
        status = "warning"

    # sequence counts
    seq_counts = df["seq_group"].fillna("UNKNOWN").value_counts().to_dict()
    for seq, count in seq_counts.items():
        if count < cfg["self_audit"]["min_cases_per_target"]:
            findings.append({"level": "warning", "item": "low_target_count", "seq_group": seq, "count": int(count)})

    # vendor missingness
    if "vendor" in df.columns:
        missing_rate = float((df["vendor"].isna() | (df["vendor"].astype(str).str.len() == 0)).mean())
        if missing_rate > cfg["self_audit"]["max_vendor_missing_rate"]:
            findings.append({"level": "warning", "item": "vendor_missing_rate_high", "value": missing_rate})

    # sequence ambiguity
    unknown_seq = int(df["seq_group"].isna().sum()) if "seq_group" in df.columns else df.shape[0]
    if unknown_seq > 0:
        findings.append({"level": "warning", "item": "unknown_sequence_group", "count": unknown_seq})

    summary = {
        "status": status,
        "n_rows": int(df.shape[0]),
        "n_patients": int(df["patient_id"].nunique()),
        "subset_counts": df["subset"].fillna("UNKNOWN").value_counts().to_dict(),
        "cohort_counts": df["cohort_role"].fillna("UNKNOWN").value_counts().to_dict(),
        "sequence_counts": seq_counts,
        "findings": findings,
    }

    with open(out_dir / "self_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown report
    lines = [
        "# Self Audit Report",
        "",
        f"- Overall status: **{status.upper()}**",
        f"- Total rows: {summary['n_rows']}",
        f"- Total patients: {summary['n_patients']}",
        "",
        "## Sequence counts",
    ]
    for seq, count in seq_counts.items():
        lines.append(f"- {seq}: {count}")
    lines += ["", "## Findings"]
    for item in findings:
        lines.append(f"- [{item['level'].upper()}] {item}")
    (out_dir / "SELF_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
