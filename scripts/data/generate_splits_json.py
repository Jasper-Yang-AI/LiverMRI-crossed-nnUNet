from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.lib.common import expand_named_collection, load_yaml, make_case_id, resolve_experiment


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
    parser = argparse.ArgumentParser(description="Create nnUNetv2 splits_final.json with strict patient-level fold assignment.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/roi_study_config.yaml")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--seq-group", default=None)
    parser.add_argument("--seq-groups", nargs="+", default=None)
    parser.add_argument("--seq-bundle", default=None)
    parser.add_argument("--source-tag", default=None)
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--nnunet-preprocessed", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    source_tag, selected_groups = resolve_selected_groups(args, cfg)
    dataset_id = args.dataset_id
    if args.experiment_id:
        dataset_id = dataset_id or int(resolve_experiment(cfg, args.experiment_id)["dataset_id"])
    if dataset_id is None:
        raise ValueError("Provide `--dataset-id` or use `--experiment-id` with a dataset_id in config.")

    df = pd.read_csv(args.manifest)
    if "fold" not in df.columns:
        raise ValueError("Manifest must contain `fold` column.")

    df = df[(df["fold"] >= 0) & (df["seq_group"].isin(selected_groups))].copy()
    if df.empty:
        raise ValueError(f"No rows found for selected groups: {selected_groups}")

    def derive_case_id(row):
        case_stem = str(row["case_stem"]) if "case_stem" in row and pd.notna(row["case_stem"]) else f"{row['patient_id']}_{row['seq_raw']}"
        return make_case_id(case_stem)

    df["case_id"] = df.apply(derive_case_id, axis=1)

    splits = []
    for fold in sorted(df["fold"].unique().tolist()):
        val_cases = sorted(df[df["fold"] == fold]["case_id"].tolist())
        train_cases = sorted(df[df["fold"] != fold]["case_id"].tolist())
        splits.append({"train": train_cases, "val": val_cases})

    dataset_dir = Path(args.nnunet_preprocessed) / f"Dataset{dataset_id:03d}_LiverTumor_{source_tag}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / "splits_final.json", "w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2, ensure_ascii=False)

    print(f"Saved splits_final.json to {dataset_dir}")
    print(f"Source tag: {source_tag}")
    print(f"Included sequence groups: {selected_groups}")


if __name__ == "__main__":
    main()

