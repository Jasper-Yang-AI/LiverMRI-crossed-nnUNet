$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptRoot '..\..\..\..')).Path
Set-Location -LiteralPath $RepoRoot
$env:nnUNet_raw = Join-Path $RepoRoot 'nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $RepoRoot 'nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $RepoRoot 'nnUNet_results'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '1'
New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null
python scripts\export_nnunet_source_dataset.py --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\manifest_with_folds.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\study_config.yaml" --experiment-id U3 --nnunet-raw ".\nnUNet_raw"
python scripts\generate_splits_json.py --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\manifest_with_folds.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\study_config.yaml" --experiment-id U3 --nnunet-preprocessed ".\nnUNet_preprocessed"
nnUNetv2_plan_and_preprocess -d 317 --verify_dataset_integrity
nnUNetv2_train 317 3d_fullres 0
nnUNetv2_train 317 3d_fullres 1
nnUNetv2_train 317 3d_fullres 2
nnUNetv2_train 317 3d_fullres 3
nnUNetv2_train 317 3d_fullres 4
python scripts\export_target_test_sets.py --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\manifest_with_folds.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\study_config.yaml" --experiment-id U3 --out-dir "outputs\experiments\U3\targets"
. "outputs\experiments\U3\commands\infer_internal_cv.ps1"
. "outputs\experiments\U3\commands\infer_external_test.ps1"
python scripts\evaluate_predictions.py --evaluation-manifest "outputs\experiments\U3\evaluation_manifest.csv" --out-csv "outputs\experiments\U3\results\per_case_metrics.csv"
python scripts\aggregate_results.py --metrics "outputs\experiments\U3\results\per_case_metrics.csv" --audit "outputs\audit\self_audit_summary.json" --out-dir "outputs\experiments\U3\paper_assets"
python scripts\make_figures.py --paper-dir "outputs\experiments\U3\paper_assets"
python scripts\build_manuscript.py --paper-dir "outputs\experiments\U3\paper_assets" --template "manuscript/manuscript_template.md" --out "manuscript\manuscript_U3.md"