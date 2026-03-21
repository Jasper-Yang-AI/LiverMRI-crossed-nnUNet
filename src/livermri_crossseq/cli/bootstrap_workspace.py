from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from livermri_crossseq.config import load_yaml_config
from livermri_crossseq.data.workspace import (
    assign_patient_folds,
    attach_roi_columns,
    build_manifest,
    create_workspace_dirs,
    summarize_manifest,
    write_summary,
    write_workspace_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a cross-sequence workspace for liver MRI experiments.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root, for example D:\\Dataset003_v2_LiverTumorSeg")
    parser.add_argument("--study-config", default="configs/dataset/livermri_crossseq_dataset.yaml")
    parser.add_argument("--crossseq-config", default="configs/experiment/livermri_crossseq_fusion.yaml")
    parser.add_argument("--workspace-root", default=None, help="Override the workspace root from the cross-sequence config")
    parser.add_argument("--manifest", default=None, help="Optional pre-reviewed manifest to reuse instead of rebuilding")
    args = parser.parse_args()

    crossseq_cfg = load_yaml_config(args.crossseq_config)
    workspace_root = Path(args.workspace_root or crossseq_cfg["experiment"]["workspace_root"])
    workspace_dirs = create_workspace_dirs(workspace_root)
    dataset_root = Path(args.dataset_root)

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_df = pd.read_csv(manifest_path)
    else:
        manifest_path = workspace_dirs["manifests"] / "manifest_proposed.csv"
        manifest_df = build_manifest(dataset_root, args.study_config, manifest_path)

    folded_df = assign_patient_folds(manifest_df)
    folded_path = workspace_dirs["manifests"] / "manifest_with_folds.csv"
    folded_df.to_csv(folded_path, index=False, encoding="utf-8-sig")

    study_cfg = load_yaml_config(args.study_config)
    roi_root = dataset_root / "liver_roi_totalseg"
    if roi_root.exists():
        roi_df = attach_roi_columns(folded_df, roi_root, float(study_cfg["roi"]["dilation_mm"]))
        roi_path = workspace_dirs["manifests"] / "manifest_with_folds_liver_roi.csv"
        roi_df.to_csv(roi_path, index=False, encoding="utf-8-sig")
        active_df = roi_df
    else:
        roi_path = None
        active_df = folded_df

    extra_env = {"LIVERMRI_CROSSSEQ_CONFIG": str(Path(args.crossseq_config).resolve())}
    write_workspace_env(workspace_root / "env" / "workspace_env.ps1", workspace_dirs, extra_env=extra_env)
    summary = summarize_manifest(active_df, roi_enabled=roi_path is not None)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "manifest_path": str(manifest_path),
            "folded_manifest_path": str(folded_path),
            "roi_manifest_path": str(roi_path) if roi_path is not None else "",
            "workspace_root": str(workspace_root),
        }
    )
    write_summary(workspace_dirs["logs"] / "bootstrap_summary.json", summary)

    print(f"Workspace root: {workspace_root}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved folds: {folded_path}")
    if roi_path is not None:
        print(f"Saved manifest with liver ROI: {roi_path}")
    else:
        print("ROI directory not found under dataset root; bootstrap continued without ROI columns.")
    print(f"PowerShell env: {workspace_root / 'env' / 'workspace_env.ps1'}")


if __name__ == "__main__":
    main()
