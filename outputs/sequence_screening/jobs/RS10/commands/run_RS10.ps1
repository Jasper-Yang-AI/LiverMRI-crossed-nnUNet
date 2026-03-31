$ErrorActionPreference = "Stop"
$CommandRoot = $PSScriptRoot
$ExperimentRoot = [System.IO.Path]::GetFullPath((Join-Path $CommandRoot '..'))
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $CommandRoot "..\..\..\..\.."))
Set-Location -LiteralPath $RepoRoot
$ManifestPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv"))
$StudyConfigPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "configs\dataset\livermri_crossseq_dataset.yaml"))
$NNUNetRawPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "nnUNet_raw_sequence_screening"))
$NNUNetPreprocessedPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "nnUNet_preprocessed_sequence_screening"))
$NNUNetResultsPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "nnUNet_results_sequence_screening"))
$ExportedCaseManifestPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "nnUNet_raw_sequence_screening\Dataset410_LiverTumor_DELAY\case_manifest.csv"))
$TargetsDir = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'targets'))
$InferInternalScript = [System.IO.Path]::GetFullPath((Join-Path $CommandRoot 'infer_internal_cv.ps1'))
$InferExternalScript = [System.IO.Path]::GetFullPath((Join-Path $CommandRoot 'infer_external_test.ps1'))
$EvaluationManifestPath = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'evaluation_manifest.csv'))
$PostprocessedManifestPath = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'evaluation_manifest_postprocessed.csv'))
$PostprocessedPredRoot = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'predictions_postprocessed'))
$ResultsDir = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'results'))
$MetricsCsvPath = [System.IO.Path]::GetFullPath((Join-Path $ResultsDir 'per_case_metrics.csv'))
$ReportsDir = [System.IO.Path]::GetFullPath((Join-Path $ExperimentRoot 'reports'))
$env:nnUNet_raw = $NNUNetRawPath
$env:nnUNet_preprocessed = $NNUNetPreprocessedPath
$env:nnUNet_results = $NNUNetResultsPath
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '0'
New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null
python -m scripts.dataset.export_sequence_screening_dataset --manifest $ManifestPath --study-config $StudyConfigPath --experiment-id RS10 --nnunet-raw $NNUNetRawPath --roi-column roi_mask_dilated_path --roi-mode masked --crop-margin-mm 20.0
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 export source dataset (exit code $LASTEXITCODE)" }
python -m scripts.dataset.generate_splits_json --manifest $ManifestPath --study-config $StudyConfigPath --experiment-id RS10 --nnunet-preprocessed $NNUNetPreprocessedPath --exported-case-manifest $ExportedCaseManifestPath
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 generate splits (exit code $LASTEXITCODE)" }
nnUNetv2_plan_and_preprocess -d 410 --verify_dataset_integrity
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 plan and preprocess (exit code $LASTEXITCODE)" }
nnUNetv2_train 410 3d_fullres 0
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 train fold 0 (exit code $LASTEXITCODE)" }
nnUNetv2_train 410 3d_fullres 1
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 train fold 1 (exit code $LASTEXITCODE)" }
nnUNetv2_train 410 3d_fullres 2
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 train fold 2 (exit code $LASTEXITCODE)" }
nnUNetv2_train 410 3d_fullres 3
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 train fold 3 (exit code $LASTEXITCODE)" }
nnUNetv2_train 410 3d_fullres 4
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 train fold 4 (exit code $LASTEXITCODE)" }
python -m scripts.dataset.export_sequence_screening_targets --manifest $ManifestPath --study-config $StudyConfigPath --experiment-id RS10 --out-dir $TargetsDir --roi-column roi_mask_dilated_path --roi-mode masked --crop-margin-mm 20.0
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 export targets (exit code $LASTEXITCODE)" }
& $InferInternalScript
& $InferExternalScript
python -m scripts.eval.constrain_predictions_to_liver_roi --evaluation-manifest $EvaluationManifestPath --out-manifest $PostprocessedManifestPath --out-root $PostprocessedPredRoot --roi-column roi_mask_dilated_path --keep-original-when-missing-roi
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 constrain predictions to liver ROI (exit code $LASTEXITCODE)" }
python -m scripts.eval.evaluate_predictions --evaluation-manifest $PostprocessedManifestPath --out-csv $MetricsCsvPath
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 evaluate predictions (exit code $LASTEXITCODE)" }
python -m scripts.eval.aggregate_results --metrics $MetricsCsvPath --out-dir $ReportsDir
if ($LASTEXITCODE -ne 0) { throw "Step failed: RS10 aggregate results (exit code $LASTEXITCODE)" }
