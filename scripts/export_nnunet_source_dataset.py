from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ensure_dir, expand_named_collection, load_yaml, make_case_id, materialize_file, resolve_experiment


def resolve_selected_groups(args, cfg) -> tuple[str, list[str]]:
    if args.experiment_id:
        exp = resolve_experiment(cfg, args.experiment_id)
        source_bundle = exp["source_bundle"]
        selected_groups = expand_named_collection(source_bundle, cfg, sections=("sequence_bundles", "target_bundles"))
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
    parser = argparse.ArgumentParser(description="Export source data into nnUNet_raw format.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--experiment-id", default=None, help="Experiment key from configs/study_config.yaml")
    parser.add_argument("--seq-group", default=None, help="Single canonical sequence group")
    parser.add_argument("--seq-groups", nargs="+", default=None, help="Multiple canonical sequence groups")
    parser.add_argument("--seq-bundle", default=None, help="Named bundle from configs/study_config.yaml")
    parser.add_argument("--source-tag", default=None, help="Model tag used in dataset naming and downstream outputs")
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    source_tag, selected_groups = resolve_selected_groups(args, cfg)
    dataset_id = args.dataset_id
    if args.experiment_id:
        dataset_id = dataset_id or int(resolve_experiment(cfg, args.experiment_id)["dataset_id"])
    if dataset_id is None:
        raise ValueError("Provide `--dataset-id` or use `--experiment-id` with a dataset_id in config.")

    df = pd.read_csv(args.manifest)
    if "fold" in df.columns:
        df = df[df["fold"] >= 0].copy()
    if "cohort_role" in df.columns:
        df = df[df["cohort_role"] == "internal_cv"].copy()

    df = df[df["seq_group"].isin(selected_groups)].copy()
    if df.empty:
        raise ValueError(f"No rows found for selected groups: {selected_groups}")

    dataset_name = f"Dataset{dataset_id:03d}_LiverTumor_{source_tag}"
    base = Path(args.nnunet_raw) / dataset_name
    images_tr = ensure_dir(base / "imagesTr")
    labels_tr = ensure_dir(base / "labelsTr")

    exported_rows = []
    for _, row in df.iterrows():
        case_stem = str(row["case_stem"]) if "case_stem" in row and pd.notna(row["case_stem"]) else f"{row['patient_id']}_{row['seq_raw']}"
        case_id = make_case_id(case_stem)
        src_img = Path(row["image_path"])
        src_lab = Path(row["label_path"])

        dst_img = images_tr / f"{case_id}_0000.nii.gz"
        dst_lab = labels_tr / f"{case_id}.nii.gz"

        materialize_file(src_img, dst_img, args.copy_mode)
        materialize_file(src_lab, dst_lab, args.copy_mode)

        exported_rows.append(
            {
                "case_id": case_id,
                "case_stem": case_stem,
                "patient_id": row["patient_id"],
                "seq_raw": row["seq_raw"],
                "seq_group": row["seq_group"],
                "fold": int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else -1,
                "subset": row.get("subset", "train"),
                "image_path": str(src_img),
                "label_path": str(src_lab),
            }
        )

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": cfg["labels"],
        "numTraining": len(exported_rows),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": f"Source export for {source_tag}",
    }

    with open(base / "dataset.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2, ensure_ascii=False)

    pd.DataFrame(exported_rows).to_csv(base / "case_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Exported {len(exported_rows)} cases to {base}")
    print(f"Source tag: {source_tag}")
    print(f"Included sequence groups: {selected_groups}")


if __name__ == "__main__":
    main()
