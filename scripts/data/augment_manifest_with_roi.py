from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.lib.common import make_case_id
from scripts.lib.roi_utils import mm_tag, sanitize_subset_name


def main():
    parser = argparse.ArgumentParser(description="Attach liver ROI paths to an existing case manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--roi-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dilation-mm", type=float, default=12.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    roi_dir = Path(args.roi_dir)
    dilation_dirname = f"dilated_{mm_tag(args.dilation_mm)}"
    existing_cols = [
        "case_id",
        "roi_mask_raw_path",
        "roi_mask_clean_path",
        "roi_mask_dilated_path",
        "roi_available",
    ]
    manifest = manifest.drop(columns=[col for col in existing_cols if col in manifest.columns], errors="ignore")

    def build_case_paths(row: pd.Series) -> pd.Series:
        case_id = make_case_id(str(row["case_stem"]))
        subset_name = sanitize_subset_name(row.get("subset", "unknown"))
        clean_path = roi_dir / "clean" / subset_name / f"{case_id}.nii.gz"
        dilated_path = roi_dir / dilation_dirname / subset_name / f"{case_id}.nii.gz"
        raw_path = roi_dir / "raw" / subset_name / f"{case_id}.nii.gz"
        return pd.Series(
            {
                "case_id": case_id,
                "roi_mask_raw_path": str(raw_path),
                "roi_mask_clean_path": str(clean_path),
                "roi_mask_dilated_path": str(dilated_path),
                "roi_available": int(clean_path.exists() and dilated_path.exists()),
            }
        )

    manifest = pd.concat([manifest, manifest.apply(build_case_paths, axis=1)], axis=1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved manifest with ROI columns to: {out_path}")
    print(f"ROI available rows: {int(manifest['roi_available'].sum())} / {len(manifest)}")


if __name__ == "__main__":
    main()

