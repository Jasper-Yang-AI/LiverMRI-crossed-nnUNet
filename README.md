# LiverMRI-CrossSeq-nnUNetv2

A publication-oriented, patient-level, cross-sequence nnUNetv2 pipeline for **real-world heterogeneous liver MRI tumor segmentation**.

This repository is designed for the study pattern discussed in the planning phase:

1. **Choose one dominant source sequence** as the main training domain (default: `T2`).
2. **Train nnUNetv2 with strict patient-level 5-fold CV** on the source sequence only.
3. **Evaluate zero-shot cross-sequence generalization** on other plain and enhanced MRI sequences.
4. **Optionally perform few-shot target-domain adaptation** on 1–2 important enhanced target sequences.
5. **Automatically aggregate metrics, generate paper-ready tables/figures, and draft a manuscript skeleton**.

## Default study framing

### Task
- Main task: **liver tumor segmentation**
- Default label map:
  - `0`: background
  - `1`: tumor

> If your labels contain both liver and tumor, edit `configs/study_config.yaml` and `configs/label_map.yaml`.

### Default sequence normalization
- Plain: `T2`, `T2WI`, `DWI`, `ADC`, `T1`, `InPhase`, `OutPhase`, `C-pre`
- Enhanced:
  - Arterial: `C+A`, `AP`
  - Portal venous: `C+V`, `VP`
  - Delayed / hepatobiliary-like delayed bucket: `C+Delay`, `HP`

## Recommended minimal publishable experiment

### Module A — Cohort audit
- Normalize sequence naming
- Build a reviewed manifest
- Check label integrity, patient leakage, fold balance, and optional vendor imbalance

### Module B — Main source-sequence baseline
- Source sequence: **T2**
- Patient-level 5-fold CV
- Model: **nnUNetv2 3d_fullres**
- Metrics:
  - Dice
  - HD95
  - Lesion recall
  - Lesion precision
  - Lesion F1
  - Volume relative error (optional)

### Module C — Zero-shot cross-sequence generalization
Apply the T2-trained model directly to:
- `T2WI`
- `DWI`
- `ADC`
- `C-pre`
- `ARTERIAL` (`C+A/AP`)
- `PORTAL` (`C+V/VP`)
- `DELAY` (`C+Delay/HP`)
- optional exploratory targets: `T1`, `InPhase`, `OutPhase`

### Module D — Few-shot adaptation (optional but recommended)
Pick 1–2 clinically important target sequences:
- recommended: `PORTAL` and/or `ARTERIAL`

Compare:
- zero-shot transfer
- few-shot fine-tuning with small target-domain subsets (e.g. 10%, 20%)

## Repository layout

```text
LiverMRI-CrossSeq-nnUNetv2/
├── configs/
├── docs/
├── manuscript/
├── outputs/
├── scripts/
├── templates/
├── run_full_study.ps1
├── run_full_study.sh
└── requirements.txt
```

## Quick start

### 1. Create environment

```bash
conda create -n liver_crossseq python=3.10 -y
conda activate liver_crossseq
pip install -r requirements.txt
```

Install nnUNetv2 as needed in your environment.

### 2. Propose and review a manifest

```bash
python scripts/propose_manifest.py \
  --root "D:\Dataset003_v2_LiverTumorSeg" \
  --out outputs/manifest_proposed.csv
```

Then manually review:
- `patient_id`
- `seq_raw`
- `seq_group`
- `image_path`
- `label_path`
- optional `vendor`, `field_strength`

### 3. Validate manifest

```bash
python scripts/validate_manifest.py \
  --manifest outputs/manifest_reviewed.csv \
  --study-config configs/study_config.yaml \
  --out-dir outputs/audit
```

### 4. Create patient-level folds

```bash
python scripts/assign_folds.py \
  --manifest outputs/manifest_reviewed.csv \
  --n-folds 5 \
  --seed 3407 \
  --out outputs/manifest_with_folds.csv
```

### 5. Export source dataset for nnUNetv2

```bash
python scripts/export_nnunet_source_dataset.py \
  --manifest outputs/manifest_with_folds.csv \
  --study-config configs/study_config.yaml \
  --dataset-id 301 \
  --seq-group T2 \
  --nnunet-raw "D:\nnUNet_raw"
```

### 6. Generate `splits_final.json`

```bash
python scripts/generate_splits_json.py \
  --manifest outputs/manifest_with_folds.csv \
  --seq-group T2 \
  --dataset-id 301 \
  --nnunet-preprocessed "D:\nnUNet_preprocessed"
```

### 7. Run nnUNetv2 training

```bash
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity
nnUNetv2_train 301 3d_fullres 0
nnUNetv2_train 301 3d_fullres 1
nnUNetv2_train 301 3d_fullres 2
nnUNetv2_train 301 3d_fullres 3
nnUNetv2_train 301 3d_fullres 4
```

### 8. Export target test sets and run cross-sequence inference

```bash
python scripts/export_target_test_sets.py \
  --manifest outputs/manifest_with_folds.csv \
  --study-config configs/study_config.yaml \
  --source-seq T2 \
  --targets T2WI DWI ADC C-pre ARTERIAL PORTAL DELAY \
  --out-dir outputs/targets
```

Then run the generated command file:
- `outputs/commands/infer_targets.ps1`
- `outputs/commands/infer_targets.sh`

### 9. Evaluate and aggregate results

```bash
python scripts/evaluate_predictions.py \
  --manifest outputs/manifest_with_folds.csv \
  --pred-root outputs/predictions \
  --out-csv outputs/results/per_case_metrics.csv

python scripts/aggregate_results.py \
  --metrics outputs/results/per_case_metrics.csv \
  --audit outputs/audit/self_audit_summary.json \
  --out-dir outputs/paper_assets
```

### 10. Generate figures and manuscript draft

```bash
python scripts/make_figures.py \
  --paper-dir outputs/paper_assets

python scripts/build_manuscript.py \
  --paper-dir outputs/paper_assets \
  --template manuscript/manuscript_template.md \
  --out manuscript/manuscript_draft.md
```

## What this repo already solves
- strict **patient-level** folds
- configurable sequence normalization
- zero-shot source→target evaluation design
- few-shot adaptation data splits
- lesion-level metrics
- auto-generated paper tables / figures
- objective self-audit report

## What still depends on your local machine / data
- real data manifest review
- actual nnUNetv2 training and inference
- optional vendor metadata filling
- final figure polishing for publication

## Suggested paper title
**Cross-sequence generalization of nnUNetv2 for real-world heterogeneous liver MRI tumor segmentation: a patient-level retrospective study**

## Suggested paper innovation statement
1. A source-sequence-based evaluation framework for real-world heterogeneous liver MRI tumor segmentation.
2. Systematic comparison of zero-shot generalization from a dominant plain MRI source to heterogeneous plain and enhanced target sequences.
3. Objective quantification of transferability, domain shift, few-shot adaptation gain, and practical limitations under strict patient-level partitioning.
