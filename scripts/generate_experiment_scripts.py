from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_dir, expand_named_collection, load_yaml, resolve_experiment


def quote_ps(value: str) -> str:
    return f'"{value}"'


def quote_sh(value: str) -> str:
    return f'"{value}"'


def build_run_lines(experiment_id: str, exp: dict, args) -> tuple[list[str], list[str]]:
    source_tag = exp.get("source_tag", experiment_id)
    dataset_id = int(exp["dataset_id"])
    target_expr = " ".join(exp.get("targets", []))

    exp_root_win = Path(args.out_dir) / experiment_id
    targets_dir_win = exp_root_win / "targets"
    results_dir_win = exp_root_win / "results"
    paper_dir_win = exp_root_win / "paper_assets"
    manuscript_path_win = Path(args.manuscript_dir) / f"manuscript_{experiment_id}.md"

    common_py = f"python scripts"
    ps_lines = [
        "$ErrorActionPreference = \"Stop\"",
        "$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "$RepoRoot = (Resolve-Path (Join-Path $ScriptRoot '..\\..\\..\\..')).Path",
        "Set-Location -LiteralPath $RepoRoot",
        "$env:nnUNet_raw = Join-Path $RepoRoot 'nnUNet_raw'",
        "$env:nnUNet_preprocessed = Join-Path $RepoRoot 'nnUNet_preprocessed'",
        "$env:nnUNet_results = Join-Path $RepoRoot 'nnUNet_results'",
        "$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'",
        f"$env:CUDA_VISIBLE_DEVICES = '{args.gpu_id}'",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null",
        "New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null",
        f"{common_py}\\export_nnunet_source_dataset.py --manifest {quote_ps(args.manifest)} --study-config {quote_ps(args.study_config)} --experiment-id {experiment_id} --nnunet-raw {quote_ps(args.nnunet_raw)}",
        f"{common_py}\\generate_splits_json.py --manifest {quote_ps(args.manifest)} --study-config {quote_ps(args.study_config)} --experiment-id {experiment_id} --nnunet-preprocessed {quote_ps(args.nnunet_preprocessed)}",
        f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity",
        f"nnUNetv2_train {dataset_id} 3d_fullres 0",
        f"nnUNetv2_train {dataset_id} 3d_fullres 1",
        f"nnUNetv2_train {dataset_id} 3d_fullres 2",
        f"nnUNetv2_train {dataset_id} 3d_fullres 3",
        f"nnUNetv2_train {dataset_id} 3d_fullres 4",
        f"{common_py}\\export_target_test_sets.py --manifest {quote_ps(args.manifest)} --study-config {quote_ps(args.study_config)} --experiment-id {experiment_id} --out-dir {quote_ps(str(targets_dir_win))}",
        f". {quote_ps(str(exp_root_win / 'commands' / 'infer_internal_cv.ps1'))}",
        f". {quote_ps(str(exp_root_win / 'commands' / 'infer_external_test.ps1'))}",
        f"{common_py}\\evaluate_predictions.py --evaluation-manifest {quote_ps(str(exp_root_win / 'evaluation_manifest.csv'))} --out-csv {quote_ps(str(results_dir_win / 'per_case_metrics.csv'))}",
        f"{common_py}\\aggregate_results.py --metrics {quote_ps(str(results_dir_win / 'per_case_metrics.csv'))} --audit {quote_ps(str(Path(args.audit_dir) / 'self_audit_summary.json'))} --out-dir {quote_ps(str(paper_dir_win))}",
        f"{common_py}\\make_figures.py --paper-dir {quote_ps(str(paper_dir_win))}",
        f"{common_py}\\build_manuscript.py --paper-dir {quote_ps(str(paper_dir_win))} --template {quote_ps(args.template)} --out {quote_ps(str(manuscript_path_win))}",
    ]

    exp_root_sh = Path(args.out_dir) / experiment_id
    targets_dir_sh = exp_root_sh / "targets"
    results_dir_sh = exp_root_sh / "results"
    paper_dir_sh = exp_root_sh / "paper_assets"
    manuscript_path_sh = Path(args.manuscript_dir) / f"manuscript_{experiment_id}.md"

    sh_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "REPO_ROOT=\"$(cd \"${SCRIPT_DIR}/../../../..\" && pwd)\"",
        "cd \"${REPO_ROOT}\"",
        "export nnUNet_raw=\"${REPO_ROOT}/nnUNet_raw\"",
        "export nnUNet_preprocessed=\"${REPO_ROOT}/nnUNet_preprocessed\"",
        "export nnUNet_results=\"${REPO_ROOT}/nnUNet_results\"",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"export CUDA_VISIBLE_DEVICES={args.gpu_id}",
        "mkdir -p \"${nnUNet_raw}\" \"${nnUNet_preprocessed}\" \"${nnUNet_results}\"",
        f"python scripts/export_nnunet_source_dataset.py --manifest {quote_sh(args.manifest)} --study-config {quote_sh(args.study_config)} --experiment-id {experiment_id} --nnunet-raw {quote_sh(args.nnunet_raw)}",
        f"python scripts/generate_splits_json.py --manifest {quote_sh(args.manifest)} --study-config {quote_sh(args.study_config)} --experiment-id {experiment_id} --nnunet-preprocessed {quote_sh(args.nnunet_preprocessed)}",
        f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity",
        f"nnUNetv2_train {dataset_id} 3d_fullres 0",
        f"nnUNetv2_train {dataset_id} 3d_fullres 1",
        f"nnUNetv2_train {dataset_id} 3d_fullres 2",
        f"nnUNetv2_train {dataset_id} 3d_fullres 3",
        f"nnUNetv2_train {dataset_id} 3d_fullres 4",
        f"python scripts/export_target_test_sets.py --manifest {quote_sh(args.manifest)} --study-config {quote_sh(args.study_config)} --experiment-id {experiment_id} --out-dir {quote_sh(str(targets_dir_sh))}",
        f"bash {quote_sh(str(exp_root_sh / 'commands' / 'infer_internal_cv.sh'))}",
        f"bash {quote_sh(str(exp_root_sh / 'commands' / 'infer_external_test.sh'))}",
        f"python scripts/evaluate_predictions.py --evaluation-manifest {quote_sh(str(exp_root_sh / 'evaluation_manifest.csv'))} --out-csv {quote_sh(str(results_dir_sh / 'per_case_metrics.csv'))}",
        f"python scripts/aggregate_results.py --metrics {quote_sh(str(results_dir_sh / 'per_case_metrics.csv'))} --audit {quote_sh(str(Path(args.audit_dir) / 'self_audit_summary.json'))} --out-dir {quote_sh(str(paper_dir_sh))}",
        f"python scripts/make_figures.py --paper-dir {quote_sh(str(paper_dir_sh))}",
        f"python scripts/build_manuscript.py --paper-dir {quote_sh(str(paper_dir_sh))} --template {quote_sh(args.template)} --out {quote_sh(str(manuscript_path_sh))}",
    ]

    return ps_lines, sh_lines


def main():
    parser = argparse.ArgumentParser(description="Generate runnable command files for all configured experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--nnunet-preprocessed", required=True)
    parser.add_argument("--audit-dir", default="outputs/audit")
    parser.add_argument("--template", default="manuscript/manuscript_template.md")
    parser.add_argument("--out-dir", default="outputs/experiments")
    parser.add_argument("--manuscript-dir", default="manuscript")
    parser.add_argument("--gpu-id", default="1", help="CUDA visible GPU id for generated run scripts")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(args.manuscript_dir)

    rows = []
    lines = [
        "# Experiment Suite",
        "",
        "This file is auto-generated from configs/study_config.yaml.",
        "",
    ]

    for experiment_id in cfg.get("experiments", {}):
        exp = resolve_experiment(cfg, experiment_id)
        source_members = expand_named_collection(exp["source_bundle"], cfg, sections=("sequence_bundles", "target_bundles"))
        targets = list(exp.get("targets", []))
        dataset_id = int(exp["dataset_id"])

        exp_dir = ensure_dir(out_dir / experiment_id)
        commands_dir = ensure_dir(exp_dir / "suite_commands")

        ps_lines, sh_lines = build_run_lines(experiment_id, exp, args)
        (commands_dir / f"run_{experiment_id}.ps1").write_text("\n".join(ps_lines), encoding="utf-8")
        (commands_dir / f"run_{experiment_id}.sh").write_text("\n".join(sh_lines), encoding="utf-8")

        rows.append(
            {
                "experiment_id": experiment_id,
                "dataset_id": dataset_id,
                "source_tag": exp.get("source_tag", experiment_id),
                "source_bundle": exp["source_bundle"],
                "source_members": "+".join(source_members),
                "targets": "+".join(targets),
                "description": exp.get("description", ""),
                "script_ps1": str(commands_dir / f"run_{experiment_id}.ps1"),
            }
        )

        lines.extend(
            [
                f"## {experiment_id}",
                "",
                f"- dataset_id: {dataset_id}",
                f"- source_bundle: {exp['source_bundle']}",
                f"- source_members: {', '.join(source_members)}",
                f"- targets: {', '.join(targets)}",
                f"- description: {exp.get('description', '')}",
                f"- run script: {commands_dir / f'run_{experiment_id}.ps1'}",
                "",
            ]
        )

    pd.DataFrame(rows).to_csv(out_dir / "experiment_registry.csv", index=False, encoding="utf-8-sig")
    (out_dir / "EXPERIMENT_SUITE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved experiment suite to: {out_dir}")


if __name__ == "__main__":
    main()
