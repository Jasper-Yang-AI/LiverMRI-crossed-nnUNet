from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.lib.common import (
    ensure_dir,
    expand_named_collection,
    get_canonical_sequences,
    load_yaml,
    make_case_id,
    resolve_experiment,
)
from scripts.lib.roi_utils import save_nifti, transform_case_with_roi


def export_cases(
    rows_df: pd.DataFrame,
    export_dir: Path,
    prediction_dir: Path,
    eval_mode: str,
    source_tag: str,
    source_primary_target: str,
    source_target_set: str,
    target_seq: str,
    fold: int,
    roi_mode: str,
    roi_column: str,
    outside_value: float,
    crop_margin_mm: float,
    clip_label: bool,
) -> list[dict]:
    images_ts = ensure_dir(export_dir / "imagesTs")
    labels_ts = ensure_dir(export_dir / "labelsTs")
    exported_rows = []

    for _, row in rows_df.iterrows():
        case_stem = str(row["case_stem"])
        case_id = make_case_id(case_stem)
        dst_img = images_ts / f"{case_id}_0000.nii.gz"
        dst_lab = labels_ts / f"{case_id}.nii.gz"

        image_nifti, label_nifti, meta = transform_case_with_roi(
            image_path=row["image_path"],
            label_path=row["label_path"],
            roi_path=row[roi_column],
            roi_mode=roi_mode,
            outside_value=outside_value,
            crop_margin_mm=crop_margin_mm,
            clip_label=clip_label,
        )
        save_nifti(image_nifti, dst_img)
        save_nifti(label_nifti, dst_lab)

        exported_rows.append(
            {
                "eval_mode": eval_mode,
                "source_seq": source_tag,
                "source_primary_target": source_primary_target,
                "source_target_set": source_target_set,
                "target_seq": target_seq,
                "raw_seq_group": row["seq_group"],
                "fold": fold,
                "subset": row.get("subset", ""),
                "cohort_role": row.get("cohort_role", ""),
                "patient_id": row["patient_id"],
                "seq_raw": row["seq_raw"],
                "seq_group": row["seq_group"],
                "case_stem": case_stem,
                "case_id": case_id,
                "image_path": str(row["image_path"]),
                "label_path": str(row["label_path"]),
                "prediction_path": str(prediction_dir / f"{case_id}.nii.gz"),
                "roi_mode": roi_mode,
                "roi_column": roi_column,
                "roi_mask_raw_path": row.get("roi_mask_raw_path", ""),
                "roi_mask_clean_path": row.get("roi_mask_clean_path", ""),
                "roi_mask_dilated_path": row.get("roi_mask_dilated_path", ""),
                **meta,
            }
        )

    pd.DataFrame(exported_rows).to_csv(export_dir / "target_manifest.csv", index=False, encoding="utf-8-sig")
    return exported_rows


def resolve_target_specs(target_names: list[str], cfg: dict) -> list[dict]:
    canonical = set(get_canonical_sequences(cfg))
    target_specs = []
    for target_name in target_names:
        if target_name in canonical:
            members = [target_name]
        else:
            members = expand_named_collection(target_name, cfg, sections=("target_bundles", "sequence_bundles"))
        target_specs.append({"target_tag": target_name, "members": members})
    return target_specs


def main():
    parser = argparse.ArgumentParser(description="Export ROI-constrained target test sets for leakage-free inference.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/roi_study_config.yaml")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--source-tag", required=False, default=None)
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pred-root", default=None)
    parser.add_argument("--modes", nargs="+", choices=["internal_cv", "external_test"], default=["internal_cv", "external_test"])
    parser.add_argument("--train-subset-values", nargs="+", default=["train", "imagesTr"])
    parser.add_argument("--external-subset-values", nargs="+", default=["test", "imagesTs"])
    parser.add_argument("--roi-column", default=None)
    parser.add_argument("--roi-mode", choices=["masked", "cropped"], default=None)
    parser.add_argument("--outside-value", type=float, default=0.0)
    parser.add_argument("--crop-margin-mm", type=float, default=None)
    parser.add_argument("--disable-label-clipping", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    roi_cfg = cfg.get("roi", {})
    dataset_id = args.dataset_id
    source_tag = args.source_tag
    target_names = args.targets
    source_primary_target = args.source_tag or ""
    source_target_names: list[str] = []

    if args.experiment_id:
        exp = resolve_experiment(cfg, args.experiment_id)
        dataset_id = dataset_id or int(exp["dataset_id"])
        source_tag = source_tag or exp.get("source_tag") or args.experiment_id
        target_names = target_names or list(exp.get("targets", []))
        source_primary_target = exp.get("primary_target", source_primary_target)
        source_target_names = list(exp.get("source_targets", []))

    if dataset_id is None:
        raise ValueError("Provide `--dataset-id` or use `--experiment-id` with a dataset_id in config.")
    if not source_tag:
        raise ValueError("Provide `--source-tag` or use `--experiment-id` with a source_tag in config.")
    if not target_names:
        raise ValueError("Provide `--targets` or use `--experiment-id` with targets in config.")
    if not source_target_names and source_primary_target:
        source_target_names = [source_primary_target]

    roi_column = args.roi_column or roi_cfg.get("screening_roi_column", "roi_mask_dilated_path")
    roi_mode = args.roi_mode or roi_cfg.get("screening_mode", "masked")
    crop_margin_mm = float(args.crop_margin_mm if args.crop_margin_mm is not None else roi_cfg.get("crop_margin_mm", 20.0))
    clip_label = not args.disable_label_clipping
    source_target_set = "+".join(source_target_names)
    target_specs = resolve_target_specs(list(target_names), cfg)

    df = pd.read_csv(args.manifest)
    if roi_column not in df.columns:
        raise ValueError(f"Manifest is missing ROI column: {roi_column}")
    out_dir = ensure_dir(args.out_dir)
    pred_root = Path(args.pred_root) if args.pred_root else ensure_dir(out_dir.parent / "predictions")

    if "subset" not in df.columns:
        df["subset"] = "train"
    if "cohort_role" not in df.columns:
        df["cohort_role"] = df["subset"].astype(str).apply(
            lambda subset: "internal_cv" if subset in args.train_subset_values else "external_test"
        )
    if "fold" not in df.columns:
        df["fold"] = -1

    df = df[df[roi_column].fillna("").astype(str).str.len() > 0].copy()
    internal_df = df[
        (df["cohort_role"] == "internal_cv")
        & (df["subset"].astype(str).isin(args.train_subset_values))
        & (df["fold"] >= 0)
    ].copy()
    external_df = df[
        (df["cohort_role"] == "external_test")
        | (df["subset"].astype(str).isin(args.external_subset_values))
        | (df["fold"] < 0)
    ].copy()

    all_export_rows: list[dict] = []
    command_blocks_ps = {"internal_cv": [], "external_test": []}

    for target_spec in target_specs:
        target = target_spec["target_tag"]
        target_members = target_spec["members"]
        if "internal_cv" in args.modes:
            target_internal = internal_df[internal_df["seq_group"].isin(target_members)].copy()
            for fold in sorted(target_internal["fold"].unique().tolist()):
                fold_rows = target_internal[target_internal["fold"] == fold].copy()
                if fold_rows.empty:
                    continue

                export_dir = ensure_dir(out_dir / "internal_cv" / target / f"fold_{fold}")
                prediction_dir = ensure_dir(pred_root / "internal_cv" / f"{source_tag}_to_{target}" / f"fold_{fold}")
                export_rows = export_cases(
                    rows_df=fold_rows,
                    export_dir=export_dir,
                    prediction_dir=prediction_dir,
                    eval_mode="internal_cv",
                    source_tag=source_tag,
                    source_primary_target=source_primary_target,
                    source_target_set=source_target_set,
                    target_seq=target,
                    fold=int(fold),
                    roi_mode=roi_mode,
                    roi_column=roi_column,
                    outside_value=args.outside_value,
                    crop_margin_mm=crop_margin_mm,
                    clip_label=clip_label,
                )
                all_export_rows.extend(export_rows)
                images_ts = export_dir / "imagesTs"
                cmd = (
                    f'nnUNetv2_predict -i "{images_ts}" -o "{prediction_dir}" '
                    f'-d {dataset_id} -c {cfg["nnunet"]["configuration"]} -f {int(fold)}'
                )
                command_blocks_ps["internal_cv"].append(cmd)

        if "external_test" in args.modes:
            target_external = external_df[external_df["seq_group"].isin(target_members)].copy()
            if not target_external.empty:
                export_dir = ensure_dir(out_dir / "external_test" / target)
                prediction_dir = ensure_dir(pred_root / "external_test" / f"{source_tag}_to_{target}")
                export_rows = export_cases(
                    rows_df=target_external,
                    export_dir=export_dir,
                    prediction_dir=prediction_dir,
                    eval_mode="external_test",
                    source_tag=source_tag,
                    source_primary_target=source_primary_target,
                    source_target_set=source_target_set,
                    target_seq=target,
                    fold=-1,
                    roi_mode=roi_mode,
                    roi_column=roi_column,
                    outside_value=args.outside_value,
                    crop_margin_mm=crop_margin_mm,
                    clip_label=clip_label,
                )
                all_export_rows.extend(export_rows)

                ensemble_folds = " ".join(str(fold) for fold in cfg["nnunet"]["ensemble_folds"])
                images_ts = export_dir / "imagesTs"
                cmd = (
                    f'nnUNetv2_predict -i "{images_ts}" -o "{prediction_dir}" '
                    f'-d {dataset_id} -c {cfg["nnunet"]["configuration"]} -f {ensemble_folds}'
                )
                command_blocks_ps["external_test"].append(cmd)

    evaluation_manifest = pd.DataFrame(all_export_rows)
    evaluation_manifest_path = out_dir.parent / "evaluation_manifest.csv"
    evaluation_manifest.to_csv(evaluation_manifest_path, index=False, encoding="utf-8-sig")

    commands_dir = ensure_dir(out_dir.parent / "commands")
    combined_ps = []
    for mode in ["internal_cv", "external_test"]:
        ps_lines = command_blocks_ps[mode]
        (commands_dir / f"infer_{mode}.ps1").write_text("\n".join(ps_lines), encoding="utf-8")
        combined_ps.extend(ps_lines)

    (commands_dir / "infer_targets.ps1").write_text("\n".join(combined_ps), encoding="utf-8")

    meta = {
        "source_tag": source_tag,
        "source_primary_target": source_primary_target,
        "source_target_set": source_target_names,
        "dataset_id": dataset_id,
        "targets": [spec["target_tag"] for spec in target_specs],
        "target_specs": target_specs,
        "modes": args.modes,
        "roi_mode": roi_mode,
        "roi_column": roi_column,
    }
    with open(out_dir.parent / "targets_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"Saved target test sets to {out_dir}")
    print(f"Saved evaluation manifest to {evaluation_manifest_path}")
    print(f"Saved inference command files to {commands_dir}")


if __name__ == "__main__":
    main()

