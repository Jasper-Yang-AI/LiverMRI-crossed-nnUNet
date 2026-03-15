$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $RepoRoot

$ROOT = "D:\Dataset003_v2_LiverTumorSeg"
$CONFIG = ".\configs\roi_study_config.yaml"
$OUT_ROOT = ".\outputs\roi_study"
$ROI_DIR = Join-Path $ROOT "liver_roi_totalseg"
$MANIFEST_DIR = Join-Path $OUT_ROOT "manifests"
$NNUNET_RAW = Join-Path $RepoRoot "nnUNet_raw_roi_screening"
$NNUNET_PREPROCESSED = Join-Path $RepoRoot "nnUNet_preprocessed_roi_screening"
$NNUNET_RESULTS = Join-Path $RepoRoot "nnUNet_results_roi_screening"

New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path $MANIFEST_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_RAW | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_PREPROCESSED | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_RESULTS | Out-Null

$env:nnUNet_raw = $NNUNET_RAW
$env:nnUNet_preprocessed = $NNUNET_PREPROCESSED
$env:nnUNet_results = $NNUNET_RESULTS

& conda run -n totalseg TotalSegmentator -h *> $null

python -m scripts.data.propose_manifest --root $ROOT --study-config $CONFIG --out (Join-Path $MANIFEST_DIR "manifest_proposed.csv")
python -m scripts.data.assign_folds --manifest (Join-Path $MANIFEST_DIR "manifest_proposed.csv") --n-folds 5 --seed 3407 --out (Join-Path $MANIFEST_DIR "manifest_with_folds.csv")

python -m scripts.roi.run_totalsegmentator_liver_roi `
  --manifest (Join-Path $MANIFEST_DIR "manifest_with_folds.csv") `
  --study-config $CONFIG `
  --out-dir $ROI_DIR `
  --device gpu:1 `
  --skip-existing

python -m scripts.data.augment_manifest_with_roi `
  --manifest (Join-Path $MANIFEST_DIR "manifest_with_folds.csv") `
  --roi-dir $ROI_DIR `
  --out (Join-Path $MANIFEST_DIR "manifest_with_folds_roi.csv") `
  --dilation-mm 12

python -m scripts.experiments.generate_roi_experiment_scripts `
  --manifest (Join-Path $MANIFEST_DIR "manifest_with_folds_roi.csv") `
  --study-config $CONFIG `
  --nnunet-raw $NNUNET_RAW `
  --nnunet-preprocessed $NNUNET_PREPROCESSED `
  --nnunet-results $NNUNET_RESULTS `
  --out-dir (Join-Path $OUT_ROOT "screening\experiments")

Write-Host ""
Write-Host "ROI screening workspace is ready."
Write-Host "1. Create the TotalSegmentator env if needed:"
Write-Host "   & .\scripts\roi\create_totalseg_env.ps1"
Write-Host "2. Run one screening experiment, for example:"
Write-Host "   & .\outputs\roi_study\screening\experiments\RS08\suite_commands\run_RS08.ps1"
Write-Host "3. Or run all screening experiments:"
Write-Host "   & .\outputs\roi_study\screening\experiments\run_all_screening.ps1"
Write-Host "4. After all experiments finish, summarize anchor ranking:"
Write-Host "   python -m scripts.experiments.summarize_sequence_screening --study-config $CONFIG --experiments-root .\outputs\roi_study\screening\experiments --out-dir .\outputs\roi_study\screening\summary --require-external-for-anchor"
Write-Host "5. Prepare registration jobs after anchor selection:"
Write-Host "   python -m scripts.experiments.prepare_registration_manifest --manifest .\outputs\roi_study\manifests\manifest_with_folds_roi.csv --study-config $CONFIG --anchor-json .\outputs\roi_study\screening\summary\recommended_anchor.json --out-dir .\outputs\roi_study\registration"
