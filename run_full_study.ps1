# PowerShell study preparation entrypoint
# This script runs the shared preprocessing steps once,
# then generates runnable scripts for experiments A, M1, M2, M3, U1, U2, U3, U4.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $RepoRoot

$ROOT = "D:\Dataset003_v2_LiverTumorSeg"
$NNUNET_RAW = "D:\nnUNet_raw"
$NNUNET_PREPROCESSED = "D:\nnUNet_preprocessed"
$NNUNET_RESULTS = "D:\nnUNet_results"

$env:nnUNet_raw = $NNUNET_RAW
$env:nnUNet_preprocessed = $NNUNET_PREPROCESSED
$env:nnUNet_results = $NNUNET_RESULTS

New-Item -ItemType Directory -Force -Path $NNUNET_RAW | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_PREPROCESSED | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_RESULTS | Out-Null

python .\scripts\propose_manifest.py --root $ROOT --out .\outputs\manifest_proposed.csv
python .\scripts\validate_manifest.py --manifest .\outputs\manifest_proposed.csv --study-config .\configs\study_config.yaml --out-dir .\outputs\audit --skip-label-values
python .\scripts\profile_dataset.py --manifest .\outputs\manifest_proposed.csv --study-config .\configs\study_config.yaml --out-dir .\outputs\profile
python .\scripts\assign_folds.py --manifest .\outputs\manifest_proposed.csv --n-folds 5 --seed 3407 --out .\outputs\manifest_with_folds.csv
python .\scripts\self_audit.py --manifest .\outputs\manifest_with_folds.csv --study-config .\configs\study_config.yaml --out-dir .\outputs\audit

python .\scripts\generate_experiment_scripts.py `
  --manifest .\outputs\manifest_with_folds.csv `
  --study-config .\configs\study_config.yaml `
  --nnunet-raw $NNUNET_RAW `
  --nnunet-preprocessed $NNUNET_PREPROCESSED `
  --audit-dir .\outputs\audit `
  --out-dir .\outputs\experiments `
  --manuscript-dir .\manuscript

Write-Host ""
Write-Host "Experiment scripts generated under outputs/experiments/<experiment_id>/suite_commands/"
Write-Host "Recommended first runs:"
Write-Host "  & .\outputs\experiments\A\suite_commands\run_A.ps1"
Write-Host "  & .\outputs\experiments\M2\suite_commands\run_M2.ps1"
