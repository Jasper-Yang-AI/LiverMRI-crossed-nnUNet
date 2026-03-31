from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from scripts.common.common import ensure_dir, load_yaml, resolve_experiment


def quote_ps(value: str) -> str:
    return f'"{value}"'


def to_abs(path_like: str | Path) -> Path:
    return Path(path_like).resolve()


def path_expr_from(base_var: str, relative_path: str) -> str:
    relative_path = relative_path.replace("/", "\\")
    return f'[System.IO.Path]::GetFullPath((Join-Path {base_var} {quote_ps(relative_path)}))'


def try_relative_to(path: Path, base: Path) -> Path | None:
    try:
        return path.relative_to(base)
    except ValueError:
        return None


def repo_path_expr(path: Path, repo_root: Path) -> str:
    relative_path = try_relative_to(path, repo_root)
    if relative_path is None:
        return quote_ps(str(path))
    return path_expr_from("$RepoRoot", "." if str(relative_path) in {"", "."} else str(relative_path))


def relpath_expr(target: Path, origin: Path, origin_var: str) -> str:
    try:
        relative_path = os.path.relpath(target, origin)
    except ValueError:
        return quote_ps(str(target))
    return path_expr_from(origin_var, relative_path)


def append_checked_step(ps_lines: list[str], command: str, step_name: str) -> None:
    ps_lines.append(command)
    ps_lines.append(f'if ($LASTEXITCODE -ne 0) {{ throw "Step failed: {step_name} (exit code $LASTEXITCODE)" }}')


def build_run_lines(experiment_id: str, exp: dict, args, cfg: dict) -> list[str]:
    dataset_id = int(exp["dataset_id"])
    source_tag = exp.get("source_tag", experiment_id)
    dataset_name = f"Dataset{dataset_id:03d}_LiverTumor_{source_tag}"
    exported_case_manifest = Path(args.nnunet_raw) / dataset_name / "case_manifest.csv"

    roi_cfg = cfg.get("roi", {})
    roi_mode = roi_cfg.get("screening_mode", "masked")
    roi_column = roi_cfg.get("screening_roi_column", "roi_mask_dilated_path")
    crop_margin_mm = float(roi_cfg.get("crop_margin_mm", 20.0))
    postprocess_roi_column = roi_cfg.get("postprocess_roi_column", roi_column)

    exp_root = to_abs(Path(args.out_dir) / experiment_id)
    results_dir = exp_root / "results"
    command_root = exp_root / "commands"

    ps_lines = [
        "$ErrorActionPreference = \"Stop\"",
        "$CommandRoot = $PSScriptRoot",
        "$ExperimentRoot = [System.IO.Path]::GetFullPath((Join-Path $CommandRoot '..'))",
        f"$RepoRoot = {relpath_expr(args.repo_root, command_root, '$CommandRoot')}",
        "Set-Location -LiteralPath $RepoRoot",
        f"$ManifestPath = {repo_path_expr(args.manifest, args.repo_root)}",
        f"$StudyConfigPath = {repo_path_expr(args.study_config, args.repo_root)}",
        f"$NNUNetRawPath = {repo_path_expr(args.nnunet_raw, args.repo_root)}",
        f"$NNUNetPreprocessedPath = {repo_path_expr(args.nnunet_preprocessed, args.repo_root)}",
        f"$NNUNetResultsPath = {repo_path_expr(args.nnunet_results, args.repo_root)}",
        f"$ExportedCaseManifestPath = {repo_path_expr(exported_case_manifest, args.repo_root)}",
        "$TargetsDir = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'targets'))",
        "$InferInternalScript = [System.IO.Path]::GetFullPath((Join-Path $CommandRoot 'infer_internal_cv.ps1'))",
        "$InferExternalScript = [System.IO.Path]::GetFullPath((Join-Path $CommandRoot 'infer_external_test.ps1'))",
        "$EvaluationManifestPath = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'evaluation_manifest.csv'))",
        "$PostprocessedManifestPath = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'evaluation_manifest_postprocessed.csv'))",
        "$PostprocessedPredRoot = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'predictions_postprocessed'))",
        "$ResultsDir = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'results'))",
        "$MetricsCsvPath = [System.IO.Path]::GetFullPath((Join-Path $ResultsDir 'per_case_metrics.csv'))",
        "$ReportsDir = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'reports'))",
        "$env:nnUNet_raw = $NNUNetRawPath",
        "$env:nnUNet_preprocessed = $NNUNetPreprocessedPath",
        "$env:nnUNet_results = $NNUNetResultsPath",
        "$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'",
        f"$env:CUDA_VISIBLE_DEVICES = '{args.gpu_id}'",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null",
    ]

    append_checked_step(
        ps_lines,
        (
            f"python -m scripts.dataset.export_sequence_screening_dataset --manifest $ManifestPath "
            f"--study-config $StudyConfigPath --experiment-id {experiment_id} "
            f"--nnunet-raw $NNUNetRawPath --roi-column {roi_column} --roi-mode {roi_mode} "
            f"--crop-margin-mm {crop_margin_mm}"
        ),
        f"{experiment_id} export source dataset",
    )
    append_checked_step(
        ps_lines,
        (
            f"python -m scripts.dataset.generate_splits_json --manifest $ManifestPath "
            f"--study-config $StudyConfigPath --experiment-id {experiment_id} "
            f"--nnunet-preprocessed $NNUNetPreprocessedPath "
            f"--exported-case-manifest $ExportedCaseManifestPath"
        ),
        f"{experiment_id} generate splits",
    )
    append_checked_step(ps_lines, f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity", f"{experiment_id} plan and preprocess")
    append_checked_step(ps_lines, f"nnUNetv2_train {dataset_id} 3d_fullres 0", f"{experiment_id} train fold 0")
    append_checked_step(ps_lines, f"nnUNetv2_train {dataset_id} 3d_fullres 1", f"{experiment_id} train fold 1")
    append_checked_step(ps_lines, f"nnUNetv2_train {dataset_id} 3d_fullres 2", f"{experiment_id} train fold 2")
    append_checked_step(ps_lines, f"nnUNetv2_train {dataset_id} 3d_fullres 3", f"{experiment_id} train fold 3")
    append_checked_step(ps_lines, f"nnUNetv2_train {dataset_id} 3d_fullres 4", f"{experiment_id} train fold 4")
    append_checked_step(
        ps_lines,
        (
            f"python -m scripts.dataset.export_sequence_screening_targets --manifest $ManifestPath "
            f"--study-config $StudyConfigPath --experiment-id {experiment_id} "
            f"--out-dir $TargetsDir --roi-column {roi_column} --roi-mode {roi_mode} "
            f"--crop-margin-mm {crop_margin_mm}"
        ),
        f"{experiment_id} export targets",
    )
    ps_lines.append("& $InferInternalScript")
    ps_lines.append("& $InferExternalScript")
    append_checked_step(
        ps_lines,
        (
            "python -m scripts.eval.constrain_predictions_to_liver_roi --evaluation-manifest "
            f"$EvaluationManifestPath --out-manifest $PostprocessedManifestPath "
            f"--out-root $PostprocessedPredRoot --roi-column {postprocess_roi_column} "
            "--keep-original-when-missing-roi"
        ),
        f"{experiment_id} constrain predictions to liver ROI",
    )
    append_checked_step(
        ps_lines,
        (
            "python -m scripts.eval.evaluate_predictions --evaluation-manifest $PostprocessedManifestPath "
            "--out-csv $MetricsCsvPath"
        ),
        f"{experiment_id} evaluate predictions",
    )
    append_checked_step(
        ps_lines,
        "python -m scripts.eval.aggregate_results --metrics $MetricsCsvPath --out-dir $ReportsDir",
        f"{experiment_id} aggregate results",
    )

    return ps_lines


def main():
    parser = argparse.ArgumentParser(description="Generate runnable jobs for liver-ROI-constrained sequence screening.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/dataset/livermri_crossseq_dataset.yaml")
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--nnunet-preprocessed", required=True)
    parser.add_argument("--nnunet-results", default=None)
    parser.add_argument("--out-dir", default="outputs/sequence_screening/jobs")
    parser.add_argument("--gpu-id", default="0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.repo_root = repo_root
    args.manifest = to_abs(args.manifest)
    args.study_config = to_abs(args.study_config)
    args.nnunet_raw = to_abs(args.nnunet_raw)
    args.nnunet_preprocessed = to_abs(args.nnunet_preprocessed)
    args.nnunet_results = (
        to_abs(args.nnunet_results) if args.nnunet_results else to_abs(repo_root / "nnUNet_results_sequence_screening")
    )
    args.out_dir = to_abs(args.out_dir)

    cfg = load_yaml(args.study_config)
    out_dir = ensure_dir(args.out_dir)
    rows = []
    suite_ps_lines = [
        "$ErrorActionPreference = \"Stop\"",
        "$JobsRoot = $PSScriptRoot",
    ]

    for experiment_id, exp in cfg.get("experiments", {}).items():
        if exp.get("stage") != "screening":
            continue
        exp = resolve_experiment(cfg, experiment_id)
        exp_dir = ensure_dir(out_dir / experiment_id)
        commands_dir = ensure_dir(exp_dir / "commands")
        ps_lines = build_run_lines(experiment_id, exp, args, cfg)
        ps_path = commands_dir / f"run_{experiment_id}.ps1"
        ps_path.write_text("\n".join(ps_lines) + "\n", encoding="utf-8")
        suite_ps_lines.append(f"& {path_expr_from('$JobsRoot', str(ps_path.relative_to(out_dir)))}")

        rows.append(
            {
                "experiment_id": experiment_id,
                "dataset_id": int(exp["dataset_id"]),
                "source_tag": exp.get("source_tag", experiment_id),
                "primary_target": exp.get("primary_target", ""),
                "description": exp.get("description", ""),
                "script_ps1": str(ps_path.relative_to(repo_root)) if try_relative_to(ps_path, repo_root) is not None else str(ps_path),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "job_registry.csv", index=False, encoding="utf-8-sig")
    (out_dir / "run_all_jobs.ps1").write_text("\n".join(suite_ps_lines) + "\n", encoding="utf-8")
    print(f"Saved sequence screening jobs to: {out_dir}")


if __name__ == "__main__":
    main()
