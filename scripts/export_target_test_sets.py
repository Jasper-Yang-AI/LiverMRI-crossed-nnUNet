from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    ensure_dir,
    expand_named_collection,
    get_canonical_sequences,
    load_yaml,
    make_case_id,
    materialize_file,
    resolve_experiment,
)


def export_cases(
    rows_df: pd.DataFrame,
    export_dir: Path,
    prediction_dir: Path,
    copy_mode: str,
    eval_mode: str,
    source_tag: str,
    source_primary_target: str,
    target_seq: str,
    fold: int,
) -> list[dict]:
    images_ts = ensure_dir(export_dir / "imagesTs")
    labels_ts = ensure_dir(export_dir / "labelsTs")
    exported_rows = []

    for _, row in rows_df.iterrows():
        case_stem = str(row["case_stem"]) if "case_stem" in row and pd.notna(row["case_stem"]) else f"{row['patient_id']}_{row['seq_raw']}"
        case_id = make_case_id(case_stem)
        src_img = Path(row["image_path"])
        src_lab = Path(row["label_path"])
        dst_img = images_ts / f"{case_id}_0000.nii.gz"
        dst_lab = labels_ts / f"{case_id}.nii.gz"

        materialize_file(src_img, dst_img, copy_mode)
        materialize_file(src_lab, dst_lab, copy_mode)

        exported_rows.append(
            {
                "eval_mode": eval_mode,
                "source_seq": source_tag,
                "source_primary_target": source_primary_target,
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
                "image_path": str(src_img),
                "label_path": str(src_lab),
                "prediction_path": str(prediction_dir / f"{case_id}.nii.gz"),
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
    parser = argparse.ArgumentParser(description="Export target sequence test sets for leakage-free inference.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--experiment-id", default=None, help="Experiment key from configs/study_config.yaml")
    parser.add_argument("--source-tag", required=False, default=None, help="Model tag, for example A, M2 or U4")
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pred-root", default=None, help="Prediction output root. Defaults to sibling outputs/predictions")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["internal_cv", "external_test"],
        default=["internal_cv", "external_test"],
        help="Evaluation modes to export",
    )
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--train-subset-values", nargs="+", default=["train", "imagesTr"])
    parser.add_argument("--external-subset-values", nargs="+", default=["test", "imagesTs"])
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    dataset_id = args.dataset_id
    source_tag = args.source_tag
    target_names = args.targets
    source_primary_target = args.source_tag or ""
    if args.experiment_id:
        exp = resolve_experiment(cfg, args.experiment_id)
        dataset_id = dataset_id or int(exp["dataset_id"])
        source_tag = source_tag or exp.get("source_tag") or args.experiment_id
        target_names = target_names or list(exp.get("targets", []))
        source_primary_target = exp.get("primary_target", source_primary_target)
    if dataset_id is None:
        raise ValueError("Provide `--dataset-id` or use `--experiment-id` with a dataset_id in config.")
    if not source_tag:
        raise ValueError("Provide `--source-tag` or use `--experiment-id` with a source_tag in config.")
    if not target_names:
        raise ValueError("Provide `--targets` or use `--experiment-id` with targets in config.")

    target_specs = resolve_target_specs(list(target_names), cfg)
    df = pd.read_csv(args.manifest)
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
    command_blocks_sh = {"internal_cv": [], "external_test": []}

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
                    copy_mode=args.copy_mode,
                    eval_mode="internal_cv",
                    source_tag=source_tag,
                    source_primary_target=source_primary_target,
                    target_seq=target,
                    fold=int(fold),
                )
                all_export_rows.extend(export_rows)
                images_ts = export_dir / "imagesTs"
                cmd = (
                    f'nnUNetv2_predict -i "{images_ts}" -o "{prediction_dir}" '
                    f'-d {dataset_id} -c {cfg["nnunet"]["configuration"]} -f {int(fold)}'
                )
                command_blocks_ps["internal_cv"].append(cmd)
                command_blocks_sh["internal_cv"].append(cmd)

        if "external_test" in args.modes:
            target_external = external_df[external_df["seq_group"].isin(target_members)].copy()
            if not target_external.empty:
                export_dir = ensure_dir(out_dir / "external_test" / target)
                prediction_dir = ensure_dir(pred_root / "external_test" / f"{source_tag}_to_{target}")
                export_rows = export_cases(
                    rows_df=target_external,
                    export_dir=export_dir,
                    prediction_dir=prediction_dir,
                    copy_mode=args.copy_mode,
                    eval_mode="external_test",
                    source_tag=source_tag,
                    source_primary_target=source_primary_target,
                    target_seq=target,
                    fold=-1,
                )
                all_export_rows.extend(export_rows)

                ensemble_folds = " ".join(str(fold) for fold in cfg["nnunet"]["ensemble_folds"])
                images_ts = export_dir / "imagesTs"
                cmd = (
                    f'nnUNetv2_predict -i "{images_ts}" -o "{prediction_dir}" '
                    f'-d {dataset_id} -c {cfg["nnunet"]["configuration"]} -f {ensemble_folds}'
                )
                command_blocks_ps["external_test"].append(cmd)
                command_blocks_sh["external_test"].append(cmd)

    evaluation_manifest = pd.DataFrame(all_export_rows)
    evaluation_manifest_path = out_dir.parent / "evaluation_manifest.csv"
    evaluation_manifest.to_csv(evaluation_manifest_path, index=False, encoding="utf-8-sig")

    commands_dir = ensure_dir(out_dir.parent / "commands")
    combined_ps = []
    combined_sh = []
    for mode in ["internal_cv", "external_test"]:
        ps_lines = command_blocks_ps[mode]
        sh_lines = command_blocks_sh[mode]
        (commands_dir / f"infer_{mode}.ps1").write_text("\n".join(ps_lines), encoding="utf-8")
        (commands_dir / f"infer_{mode}.sh").write_text("\n".join(sh_lines), encoding="utf-8")
        combined_ps.extend(ps_lines)
        combined_sh.extend(sh_lines)

    (commands_dir / "infer_targets.ps1").write_text("\n".join(combined_ps), encoding="utf-8")
    (commands_dir / "infer_targets.sh").write_text("\n".join(combined_sh), encoding="utf-8")

    meta = {
        "source_tag": source_tag,
        "source_primary_target": source_primary_target,
        "dataset_id": dataset_id,
        "targets": [spec["target_tag"] for spec in target_specs],
        "target_specs": target_specs,
        "modes": args.modes,
    }
    with open(out_dir.parent / "targets_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"Saved target test sets to {out_dir}")
    print(f"Saved evaluation manifest to {evaluation_manifest_path}")
    print(f"Saved inference command files to {commands_dir}")


if __name__ == "__main__":
    main()
