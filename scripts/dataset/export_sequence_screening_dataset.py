from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.common.common import ensure_dir, expand_named_collection, load_yaml, make_case_id, resolve_experiment
from scripts.common.roi_utils import save_nifti, transform_case_with_roi


def resolve_selected_groups(args, cfg) -> tuple[str, list[str]]:
    if args.experiment_id:
        exp = resolve_experiment(cfg, args.experiment_id)
        selected_groups = expand_named_collection(exp["source_bundle"], cfg, sections=("sequence_bundles", "target_bundles"))
        source_tag = args.source_tag or exp.get("source_tag") or args.experiment_id
        return source_tag, selected_groups

    if args.seq_bundle:
        selected_groups = expand_named_collection(args.seq_bundle, cfg, sections=("sequence_bundles", "target_bundles"))
        source_tag = args.source_tag or args.seq_bundle
        return source_tag, selected_groups

    if args.seq_groups:
        selected_groups = args.seq_groups
        source_tag = args.source_tag or "_".join(selected_groups)
        return source_tag, selected_groups

    if args.seq_group:
        return args.source_tag or args.seq_group, [args.seq_group]

    raise ValueError("Provide one of `--seq-group`, `--seq-groups`, or `--seq-bundle`.")


def main():
    parser = argparse.ArgumentParser(description="Export ROI-constrained source data into nnUNet_raw format.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/dataset/livermri_crossseq_dataset.yaml")
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--seq-group", default=None)
    parser.add_argument("--seq-groups", nargs="+", default=None)
    parser.add_argument("--seq-bundle", default=None)
    parser.add_argument("--source-tag", default=None)
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--roi-column", default=None)
    parser.add_argument("--roi-mode", choices=["masked", "cropped"], default=None)
    parser.add_argument("--outside-value", type=float, default=0.0)
    parser.add_argument("--crop-margin-mm", type=float, default=None)
    parser.add_argument("--disable-label-clipping", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    roi_cfg = cfg.get("roi", {})
    source_tag, selected_groups = resolve_selected_groups(args, cfg)
    dataset_id = args.dataset_id
    if args.experiment_id:
        dataset_id = dataset_id or int(resolve_experiment(cfg, args.experiment_id)["dataset_id"])
    if dataset_id is None:
        raise ValueError("Provide `--dataset-id` or use `--experiment-id` with a dataset_id in config.")

    roi_column = args.roi_column or roi_cfg.get("screening_roi_column", "roi_mask_dilated_path")
    roi_mode = args.roi_mode or roi_cfg.get("screening_mode", "masked")
    crop_margin_mm = float(args.crop_margin_mm if args.crop_margin_mm is not None else roi_cfg.get("crop_margin_mm", 20.0))
    clip_label = not args.disable_label_clipping

    df = pd.read_csv(args.manifest)
    if "fold" in df.columns:
        df = df[df["fold"] >= 0].copy()
    if "cohort_role" in df.columns:
        df = df[df["cohort_role"] == "internal_cv"].copy()

    df = df[df["seq_group"].isin(selected_groups)].copy()
    if roi_column not in df.columns:
        raise ValueError(f"Manifest is missing ROI column: {roi_column}")
    df = df[df[roi_column].fillna("").astype(str).str.len() > 0].copy()
    if df.empty:
        raise ValueError(f"No rows found for selected groups {selected_groups} with usable ROI masks.")

    dataset_name = f"Dataset{dataset_id:03d}_LiverTumor_{source_tag}"
    base = Path(args.nnunet_raw) / dataset_name
    images_tr = ensure_dir(base / "imagesTr")
    labels_tr = ensure_dir(base / "labelsTr")

    exported_rows = []
    for _, row in df.iterrows():
        case_stem = str(row["case_stem"])
        case_id = make_case_id(case_stem)
        dst_img = images_tr / f"{case_id}_0000.nii.gz"
        dst_lab = labels_tr / f"{case_id}.nii.gz"
        image_nifti, label_nifti, meta = transform_case_with_roi(
            image_path=row["image_path"],
            label_path=row["label_path"],
            roi_path=row[roi_column],
            roi_mode=roi_mode,
            outside_value=args.outside_value,
            crop_margin_mm=crop_margin_mm,
            clip_label=clip_label,
        )
        save_nifti(image_nifti, dst_img)
        save_nifti(label_nifti, dst_lab)

        exported_rows.append(
            {
                "case_id": case_id,
                "case_stem": case_stem,
                "patient_id": row["patient_id"],
                "seq_raw": row["seq_raw"],
                "seq_group": row["seq_group"],
                "fold": int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else -1,
                "subset": row.get("subset", "train"),
                "image_path": str(row["image_path"]),
                "label_path": str(row["label_path"]),
                "roi_path": str(row[roi_column]),
                "roi_mode": roi_mode,
                "roi_column": roi_column,
                "outside_value": args.outside_value,
                **meta,
            }
        )

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": cfg["labels"],
        "numTraining": len(exported_rows),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": f"ROI-constrained source export for {source_tag} ({roi_mode})",
    }

    with open(base / "dataset.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2, ensure_ascii=False)

    pd.DataFrame(exported_rows).to_csv(base / "case_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Exported {len(exported_rows)} cases to {base}")
    print(f"Source tag: {source_tag}")
    print(f"Included sequence groups: {selected_groups}")
    print(f"ROI mode: {roi_mode} | ROI column: {roi_column}")


if __name__ == "__main__":
    main()

