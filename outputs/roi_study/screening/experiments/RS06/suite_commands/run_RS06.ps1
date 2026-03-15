$ErrorActionPreference = "Stop"
$RepoRoot = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2"
Set-Location -LiteralPath $RepoRoot
$env:nnUNet_raw = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_raw_roi_screening"
$env:nnUNet_preprocessed = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_preprocessed_roi_screening"
$env:nnUNet_results = "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_results_roi_screening"
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '0'
New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null
python -m scripts.data.export_nnunet_source_dataset_roi --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\manifests\manifest_with_folds_roi.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\roi_study_config.yaml" --experiment-id RS06 --nnunet-raw "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_raw_roi_screening" --roi-column roi_mask_dilated_path --roi-mode masked --crop-margin-mm 20.0
python -m scripts.data.generate_splits_json --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\manifests\manifest_with_folds_roi.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\roi_study_config.yaml" --experiment-id RS06 --nnunet-preprocessed "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\nnUNet_preprocessed_roi_screening"
nnUNetv2_plan_and_preprocess -d 406 --verify_dataset_integrity
nnUNetv2_train 406 3d_fullres 0
nnUNetv2_train 406 3d_fullres 1
nnUNetv2_train 406 3d_fullres 2
nnUNetv2_train 406 3d_fullres 3
nnUNetv2_train 406 3d_fullres 4
python -m scripts.data.export_target_test_sets_roi --manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\manifests\manifest_with_folds_roi.csv" --study-config "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\configs\roi_study_config.yaml" --experiment-id RS06 --out-dir "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\targets" --roi-column roi_mask_dilated_path --roi-mode masked --crop-margin-mm 20.0
& "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\commands\infer_internal_cv.ps1"
& "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\commands\infer_external_test.ps1"
python -m scripts.evaluation.postprocess_predictions_by_liver_roi --evaluation-manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\evaluation_manifest.csv" --out-manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\evaluation_manifest_postprocessed.csv" --out-root "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\predictions_postprocessed" --roi-column roi_mask_dilated_path --keep-original-when-missing-roi
python -m scripts.evaluation.evaluate_predictions --evaluation-manifest "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\evaluation_manifest_postprocessed.csv" --out-csv "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\results\per_case_metrics.csv"
python -m scripts.evaluation.aggregate_results --metrics "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\results\per_case_metrics.csv" --out-dir "D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2\outputs\roi_study\screening\experiments\RS06\paper_assets"