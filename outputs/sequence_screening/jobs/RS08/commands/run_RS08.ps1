$ErrorActionPreference = "Stop"
$RepoRoot = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2"
Set-Location -LiteralPath $RepoRoot
$env:nnUNet_raw = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_raw_sequence_screening"
$env:nnUNet_preprocessed = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_preprocessed_sequence_screening"
$env:nnUNet_results = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_results_sequence_screening"
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '0'
New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null
python -m scripts.dataset.export_sequence_screening_dataset --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\dataset\livermri_crossseq_dataset.yaml" --experiment-id RS08 --nnunet-raw "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_raw_sequence_screening" --roi-column roi_mask_dilated_path --roi-mode masked --crop-margin-mm 20.0
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 export source dataset (exit code $LASTEXITCODE)" }
python -m scripts.dataset.generate_splits_json --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\dataset\livermri_crossseq_dataset.yaml" --experiment-id RS08 --nnunet-preprocessed "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_preprocessed_sequence_screening" --exported-case-manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_raw_sequence_screening\Dataset408_LiverTumor_ARTERIAL\case_manifest.csv"
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 generate splits (exit code $LASTEXITCODE)" }
nnUNetv2_plan_and_preprocess -d 408 --verify_dataset_integrity
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 plan and preprocess (exit code $LASTEXITCODE)" }
nnUNetv2_train 408 3d_fullres 0
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 train fold 0 (exit code $LASTEXITCODE)" }
nnUNetv2_train 408 3d_fullres 1
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 train fold 1 (exit code $LASTEXITCODE)" }
nnUNetv2_train 408 3d_fullres 2
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 train fold 2 (exit code $LASTEXITCODE)" }
nnUNetv2_train 408 3d_fullres 3
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 train fold 3 (exit code $LASTEXITCODE)" }
nnUNetv2_train 408 3d_fullres 4
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 train fold 4 (exit code $LASTEXITCODE)" }
python -m scripts.dataset.export_sequence_screening_targets --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\dataset\livermri_crossseq_dataset.yaml" --experiment-id RS08 --out-dir "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\targets" --roi-column roi_mask_dilated_path --roi-mode masked --crop-margin-mm 20.0
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 export targets (exit code $LASTEXITCODE)" }
& "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\commands\infer_internal_cv.ps1"
& "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\commands\infer_external_test.ps1"
python -m scripts.eval.constrain_predictions_to_liver_roi --evaluation-manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\evaluation_manifest.csv" --out-manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\evaluation_manifest_postprocessed.csv" --out-root "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\predictions_postprocessed" --roi-column roi_mask_dilated_path --keep-original-when-missing-roi
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 constrain predictions to liver ROI (exit code $LASTEXITCODE)" }
python -m scripts.eval.evaluate_predictions --evaluation-manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\evaluation_manifest_postprocessed.csv" --out-csv "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\results\per_case_metrics.csv"
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 evaluate predictions (exit code $LASTEXITCODE)" }
python -m scripts.eval.aggregate_results --metrics "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\results\per_case_metrics.csv" --out-dir "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\sequence_screening\jobs\RS08\reports"
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS08 aggregate results (exit code $LASTEXITCODE)" }
