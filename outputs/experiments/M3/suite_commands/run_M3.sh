#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export nnUNet_raw="${REPO_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/nnUNet_results"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"
python scripts/export_nnunet_source_dataset.py --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\manifest_with_folds.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\study_config.yaml" --experiment-id M3 --nnunet-raw ".\nnUNet_raw"
python scripts/generate_splits_json.py --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\manifest_with_folds.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\study_config.yaml" --experiment-id M3 --nnunet-preprocessed ".\nnUNet_preprocessed"
nnUNetv2_plan_and_preprocess -d 314 --verify_dataset_integrity
nnUNetv2_train 314 3d_fullres 0
nnUNetv2_train 314 3d_fullres 1
nnUNetv2_train 314 3d_fullres 2
nnUNetv2_train 314 3d_fullres 3
nnUNetv2_train 314 3d_fullres 4
python scripts/export_target_test_sets.py --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\manifest_with_folds.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\study_config.yaml" --experiment-id M3 --out-dir "outputs\experiments\M3\targets"
bash "outputs\experiments\M3\commands\infer_internal_cv.sh"
bash "outputs\experiments\M3\commands\infer_external_test.sh"
python scripts/evaluate_predictions.py --evaluation-manifest "outputs\experiments\M3\evaluation_manifest.csv" --out-csv "outputs\experiments\M3\results\per_case_metrics.csv"
python scripts/aggregate_results.py --metrics "outputs\experiments\M3\results\per_case_metrics.csv" --audit "outputs\audit\self_audit_summary.json" --out-dir "outputs\experiments\M3\paper_assets"
python scripts/make_figures.py --paper-dir "outputs\experiments\M3\paper_assets"
python scripts/build_manuscript.py --paper-dir "outputs\experiments\M3\paper_assets" --template "manuscript/manuscript_template.md" --out "manuscript\manuscript_M3.md"