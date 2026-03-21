param(
  [string]$DatasetRoot = "D:\Dataset003_v2_LiverTumorSeg",
  [string]$StudyConfig = ".\configs\dataset\livermri_crossseq_dataset.yaml",
  [string]$CrossSeqConfig = ".\configs\experiment\livermri_crossseq_fusion.yaml"
)

python -m scripts.workflow.bootstrap_workspace `
  --dataset-root $DatasetRoot `
  --study-config $StudyConfig `
  --crossseq-config $CrossSeqConfig

python -m scripts.workflow.plan_anchor_pilot `
  --screening-summary .\outputs\sequence_screening\summary\sequence_screening_summary.csv `
  --manifest .\workspaces\livermri_crossseq\manifests\manifest_with_folds_liver_roi.csv `
  --study-config $StudyConfig `
  --crossseq-config $CrossSeqConfig

Write-Host ""
Write-Host "Workspace bootstrap finished."
Write-Host "Next:"
Write-Host "1. Dot-source .\workspaces\livermri_crossseq\env\workspace_env.ps1"
Write-Host "2. Run registration for the selected anchor pilot"
Write-Host "3. Export the registered multibranch dataset"
