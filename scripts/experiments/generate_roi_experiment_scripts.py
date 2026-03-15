from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.lib.common import ensure_dir, load_yaml, resolve_experiment


def quote_ps(value: str) -> str:
    return f'"{value}"'


def to_abs(path_like: str | Path) -> Path:
    return Path(path_like).resolve()


def build_run_lines(experiment_id: str, exp: dict, args, cfg: dict) -> list[str]:
    dataset_id = int(exp["dataset_id"])
    roi_cfg = cfg.get("roi", {})
    roi_mode = roi_cfg.get("screening_mode", "masked")
    roi_column = roi_cfg.get("screening_roi_column", "roi_mask_dilated_path")
    crop_margin_mm = float(roi_cfg.get("crop_margin_mm", 20.0))
    postprocess_roi_column = roi_cfg.get("postprocess_roi_column", roi_column)

    exp_root = to_abs(Path(args.out_dir) / experiment_id)
    targets_dir = exp_root / "targets"
    results_dir = exp_root / "results"
    paper_dir = exp_root / "paper_assets"
    infer_internal_ps = exp_root / "commands" / "infer_internal_cv.ps1"
    infer_external_ps = exp_root / "commands" / "infer_external_test.ps1"
    postprocessed_manifest = exp_root / "evaluation_manifest_postprocessed.csv"
    postprocessed_pred_root = exp_root / "predictions_postprocessed"

    ps_lines = [
        "$ErrorActionPreference = \"Stop\"",
        f"$RepoRoot = {quote_ps(str(args.repo_root))}",
        "Set-Location -LiteralPath $RepoRoot",
        f"$env:nnUNet_raw = {quote_ps(str(args.nnunet_raw))}",
        f"$env:nnUNet_preprocessed = {quote_ps(str(args.nnunet_preprocessed))}",
        f"$env:nnUNet_results = {quote_ps(str(args.nnunet_results))}",
        "$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'",
        f"$env:CUDA_VISIBLE_DEVICES = '{args.gpu_id}'",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null",
        (
            f"python -m scripts.data.export_nnunet_source_dataset_roi --manifest {quote_ps(str(args.manifest))} "
            f"--study-config {quote_ps(str(args.study_config))} --experiment-id {experiment_id} "
            f"--nnunet-raw {quote_ps(str(args.nnunet_raw))} --roi-column {roi_column} --roi-mode {roi_mode} "
            f"--crop-margin-mm {crop_margin_mm}"
        ),
        (
            f"python -m scripts.data.generate_splits_json --manifest {quote_ps(str(args.manifest))} "
            f"--study-config {quote_ps(str(args.study_config))} --experiment-id {experiment_id} "
            f"--nnunet-preprocessed {quote_ps(str(args.nnunet_preprocessed))}"
        ),
        f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity",
        f"nnUNetv2_train {dataset_id} 3d_fullres 0",
        f"nnUNetv2_train {dataset_id} 3d_fullres 1",
        f"nnUNetv2_train {dataset_id} 3d_fullres 2",
        f"nnUNetv2_train {dataset_id} 3d_fullres 3",
        f"nnUNetv2_train {dataset_id} 3d_fullres 4",
        (
            f"python -m scripts.data.export_target_test_sets_roi --manifest {quote_ps(str(args.manifest))} "
            f"--study-config {quote_ps(str(args.study_config))} --experiment-id {experiment_id} "
            f"--out-dir {quote_ps(str(targets_dir))} --roi-column {roi_column} --roi-mode {roi_mode} "
            f"--crop-margin-mm {crop_margin_mm}"
        ),
        f"& {quote_ps(str(infer_internal_ps))}",
        f"& {quote_ps(str(infer_external_ps))}",
        (
            f"python -m scripts.evaluation.postprocess_predictions_by_liver_roi --evaluation-manifest "
            f"{quote_ps(str(exp_root / 'evaluation_manifest.csv'))} --out-manifest {quote_ps(str(postprocessed_manifest))} "
            f"--out-root {quote_ps(str(postprocessed_pred_root))} --roi-column {postprocess_roi_column} "
            "--keep-original-when-missing-roi"
        ),
        (
            f"python -m scripts.evaluation.evaluate_predictions --evaluation-manifest {quote_ps(str(postprocessed_manifest))} "
            f"--out-csv {quote_ps(str(results_dir / 'per_case_metrics.csv'))}"
        ),
        f"python -m scripts.evaluation.aggregate_results --metrics {quote_ps(str(results_dir / 'per_case_metrics.csv'))} --out-dir {quote_ps(str(paper_dir))}",
    ]

    return ps_lines


def main():
    parser = argparse.ArgumentParser(description="Generate runnable scripts for ROI-constrained screening experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/roi_study_config.yaml")
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--nnunet-preprocessed", required=True)
    parser.add_argument("--nnunet-results", default=None)
    parser.add_argument("--out-dir", default="outputs/roi_study/screening/experiments")
    parser.add_argument("--gpu-id", default="0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.repo_root = repo_root
    args.manifest = to_abs(args.manifest)
    args.study_config = to_abs(args.study_config)
    args.nnunet_raw = to_abs(args.nnunet_raw)
    args.nnunet_preprocessed = to_abs(args.nnunet_preprocessed)
    args.nnunet_results = to_abs(args.nnunet_results) if args.nnunet_results else to_abs(repo_root / "nnUNet_results_roi_screening")
    args.out_dir = to_abs(args.out_dir)

    cfg = load_yaml(args.study_config)
    out_dir = ensure_dir(args.out_dir)
    rows = []
    suite_ps_lines = []

    for experiment_id, exp in cfg.get("experiments", {}).items():
        if exp.get("stage") != "screening":
            continue
        exp = resolve_experiment(cfg, experiment_id)
        exp_dir = ensure_dir(out_dir / experiment_id)
        commands_dir = ensure_dir(exp_dir / "suite_commands")
        ps_lines = build_run_lines(experiment_id, exp, args, cfg)
        ps_path = commands_dir / f"run_{experiment_id}.ps1"
        ps_path.write_text("\n".join(ps_lines), encoding="utf-8")
        suite_ps_lines.append(f"& {quote_ps(str(ps_path))}")

        rows.append(
            {
                "experiment_id": experiment_id,
                "dataset_id": int(exp["dataset_id"]),
                "source_tag": exp.get("source_tag", experiment_id),
                "primary_target": exp.get("primary_target", ""),
                "description": exp.get("description", ""),
                "script_ps1": str(ps_path),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "experiment_registry.csv", index=False, encoding="utf-8-sig")
    (out_dir / "run_all_screening.ps1").write_text("\n".join(suite_ps_lines), encoding="utf-8")
    print(f"Saved ROI experiment suite to: {out_dir}")


if __name__ == "__main__":
    main()

