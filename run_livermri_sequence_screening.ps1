$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $RepoRoot

$ROOT = "D:\Dataset003_v2_LiverTumorSeg"
$CONFIG = ".\configs\dataset\livermri_crossseq_dataset.yaml"
$OUT_ROOT = ".\outputs\sequence_screening"
$ROI_DIR = Join-Path $ROOT "liver_roi_totalseg"
$MANIFEST_DIR = Join-Path $OUT_ROOT "manifests"
$NNUNET_RAW = Join-Path $RepoRoot "nnUNet_raw_sequence_screening"
$NNUNET_PREPROCESSED = Join-Path $RepoRoot "nnUNet_preprocessed_sequence_screening"
$NNUNET_RESULTS = Join-Path $RepoRoot "nnUNet_results_sequence_screening"

New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path $MANIFEST_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_RAW | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_PREPROCESSED | Out-Null
New-Item -ItemType Directory -Force -Path $NNUNET_RESULTS | Out-Null

$env:nnUNet_raw = $NNUNET_RAW
$env:nnUNet_preprocessed = $NNUNET_PREPROCESSED
$env:nnUNet_results = $NNUNET_RESULTS

& conda run -n totalseg TotalSegmentator -h *> $null

python -m scripts.dataset.propose_manifest --root $ROOT --study-config $CONFIG --out (Join-Path $MANIFEST_DIR "manifest_proposed.csv")
python -m scripts.dataset.assign_folds --manifest (Join-Path $MANIFEST_DIR "manifest_proposed.csv") --n-folds 5 --seed 3407 --out (Join-Path $MANIFEST_DIR "manifest_with_folds.csv")

python -m scripts.preprocess.run_totalsegmentator_liver_roi `
  --manifest (Join-Path $MANIFEST_DIR "manifest_with_folds.csv") `
  --study-config $CONFIG `
  --out-dir $ROI_DIR `
  --device gpu:1 `
  --skip-existing

python -m scripts.dataset.attach_liver_roi_to_manifest `
  --manifest (Join-Path $MANIFEST_DIR "manifest_with_folds.csv") `
  --roi-dir $ROI_DIR `
  --out (Join-Path $MANIFEST_DIR "manifest_with_folds_liver_roi.csv") `
  --dilation-mm 12

python -m scripts.experiment.generate_sequence_screening_jobs `
  --manifest (Join-Path $MANIFEST_DIR "manifest_with_folds_liver_roi.csv") `
  --study-config $CONFIG `
  --nnunet-raw $NNUNET_RAW `
  --nnunet-preprocessed $NNUNET_PREPROCESSED `
  --nnunet-results $NNUNET_RESULTS `
  --out-dir (Join-Path $OUT_ROOT "jobs")

Write-Host ""
Write-Host "Sequence screening workspace is ready."
Write-Host "1. Create the TotalSegmentator env if needed:"
Write-Host "   Configure the totalseg env first, then rerun this script."
Write-Host "2. Run one screening experiment, for example:"
Write-Host "   & .\outputs\sequence_screening\jobs\RS08\commands\run_RS08.ps1"
Write-Host "3. Or run all screening experiments:"
Write-Host "   & .\outputs\sequence_screening\jobs\run_all_jobs.ps1"
Write-Host "4. After all experiments finish, summarize anchor ranking:"
Write-Host "   python -m scripts.experiment.summarize_sequence_screening --study-config $CONFIG --experiments-root .\outputs\sequence_screening\jobs --out-dir .\outputs\sequence_screening\summary --require-external-for-anchor"
Write-Host "5. Prepare registration jobs after anchor selection:"
Write-Host "   python -m scripts.experiment.prepare_registration_manifest --manifest .\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv --study-config $CONFIG --anchor-json .\outputs\sequence_screening\summary\recommended_anchor.json --out-dir .\outputs\sequence_screening\registration"
