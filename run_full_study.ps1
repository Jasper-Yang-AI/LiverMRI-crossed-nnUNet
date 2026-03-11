# PowerShell end-to-end command template
# Edit paths first.

$ROOT="D:\Dataset003_v2_LiverTumorSeg"
$NNUNET_RAW="D:\nnUNet_raw"
$NNUNET_PREPROCESSED="D:\nnUNet_preprocessed"

python scripts/propose_manifest.py --root $ROOT --out outputs/manifest_proposed.csv
Write-Host "Please manually review outputs/manifest_proposed.csv and save reviewed file as outputs/manifest_reviewed.csv"

python scripts/validate_manifest.py --manifest outputs/manifest_reviewed.csv --study-config configs/study_config.yaml --out-dir outputs/audit
python scripts/assign_folds.py --manifest outputs/audit/manifest_validated.csv --n-folds 5 --seed 3407 --out outputs/manifest_with_folds.csv
python scripts/self_audit.py --manifest outputs/manifest_with_folds.csv --study-config configs/study_config.yaml --out-dir outputs/audit

python scripts/export_nnunet_source_dataset.py --manifest outputs/manifest_with_folds.csv --study-config configs/study_config.yaml --dataset-id 301 --seq-group T2 --nnunet-raw $NNUNET_RAW
python scripts/generate_splits_json.py --manifest outputs/manifest_with_folds.csv --seq-group T2 --dataset-id 301 --nnunet-preprocessed $NNUNET_PREPROCESSED

nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
nnUNetv2_train 301 3d_fullres 0
nnUNetv2_train 301 3d_fullres 1
nnUNetv2_train 301 3d_fullres 2
nnUNetv2_train 301 3d_fullres 3
nnUNetv2_train 301 3d_fullres 4

python scripts/export_target_test_sets.py --manifest outputs/manifest_with_folds.csv --study-config configs/study_config.yaml --source-seq T2 --targets T2WI DWI ADC C-pre ARTERIAL PORTAL DELAY --out-dir outputs/targets
. .\outputs\commands\infer_targets.ps1

python scripts/evaluate_predictions.py --manifest outputs/manifest_with_folds.csv --pred-root outputs/predictions --out-csv outputs/results/per_case_metrics.csv
python scripts/aggregate_results.py --metrics outputs/results/per_case_metrics.csv --audit outputs/audit/self_audit_summary.json --out-dir outputs/paper_assets
python scripts/make_figures.py --paper-dir outputs/paper_assets
python scripts/build_manuscript.py --paper-dir outputs/paper_assets --template manuscript/manuscript_template.md --out manuscript/manuscript_draft.md
