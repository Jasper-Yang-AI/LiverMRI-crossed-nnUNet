from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scripts.lib.common import ensure_dir, load_yaml, normalize_sequence


def main():
    parser = argparse.ArgumentParser(description="Validate reviewed manifest for publication-grade use.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/roi_study_config.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick smoke-checks")
    parser.add_argument("--skip-label-values", action="store_true", help="Skip reading full label arrays for faster validation")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    cfg = load_yaml(args.study_config)
    aliases = cfg["sequence_aliases"]
    expected_labels = set(cfg["labels"].values())

    df = pd.read_csv(args.manifest)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
    if "subset" not in df.columns:
        df["subset"] = "train"

    required_cols = ["patient_id", "seq_raw", "image_path", "label_path"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if "seq_group" not in df.columns:
        df["seq_group"] = df["seq_raw"].apply(lambda x: normalize_sequence(str(x), aliases))
    else:
        df["seq_group"] = df["seq_group"].fillna("").replace("", np.nan)
        df["seq_group"] = df.apply(
            lambda r: r["seq_group"] if pd.notna(r["seq_group"]) else normalize_sequence(str(r["seq_raw"]), aliases),
            axis=1,
        )

    issues = []
    for idx, row in df.iterrows():
        image_path = Path(str(row["image_path"]))
        label_path = Path(str(row["label_path"])) if pd.notna(row["label_path"]) else None

        if not image_path.exists():
            issues.append({"row": idx, "issue": "missing_image", "path": str(image_path)})
            continue

        if label_path is None or not label_path.exists():
            issues.append({"row": idx, "issue": "missing_label", "path": str(label_path)})
            continue

        if cfg["self_audit"].get("check_label_values", True):
            try:
                img_obj = nib.load(str(image_path))
                lab_obj = nib.load(str(label_path))
                if img_obj.shape != lab_obj.shape:
                    issues.append(
                        {
                            "row": idx,
                            "issue": "shape_mismatch",
                            "image_shape": list(img_obj.shape),
                            "label_shape": list(lab_obj.shape),
                            "image_path": str(image_path),
                            "label_path": str(label_path),
                        }
                    )
                if not args.skip_label_values:
                    lab = lab_obj.get_fdata()
                    uniq = set(np.unique(lab).astype(int).tolist())
                    if not uniq.issubset(expected_labels):
                        issues.append(
                            {"row": idx, "issue": "unexpected_label_values", "values": sorted(list(uniq)), "path": str(label_path)}
                        )
            except Exception as e:
                issues.append({"row": idx, "issue": "label_read_error", "error": str(e), "path": str(label_path)})

        if pd.isna(row["seq_group"]):
            issues.append({"row": idx, "issue": "unknown_sequence_group", "seq_raw": row["seq_raw"]})

    # duplicate image rows
    dup_images = df[df.duplicated(subset=["image_path"], keep=False)]
    if not dup_images.empty:
        issues.append({"issue": "duplicate_image_paths", "count": int(dup_images.shape[0])})

    summary = {
        "n_rows": int(df.shape[0]),
        "n_patients": int(df["patient_id"].nunique()),
        "subset_counts": df["subset"].fillna("UNKNOWN").value_counts().to_dict(),
        "sequence_counts": df["seq_group"].fillna("UNKNOWN").value_counts().to_dict(),
        "missing_vendor_rate": float((df["vendor"].isna() | (df["vendor"].astype(str).str.len() == 0)).mean()) if "vendor" in df.columns else 1.0,
        "n_issues": len(issues),
        "max_rows": args.max_rows,
        "skip_label_values": bool(args.skip_label_values),
    }

    df.to_csv(out_dir / "manifest_validated.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(issues).to_csv(out_dir / "validation_issues.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved validated manifest and issue reports to: {out_dir}")


if __name__ == "__main__":
    main()

