# Full Experiment Plan

## Final confirmed design

### Primary question
Can a model trained only on the dominant source sequence (`T2` by default) generalize to heterogeneous real-world liver MRI tumor segmentation targets?

### Primary hypothesis
A patient-level nnUNetv2 model trained on the dominant plain MRI source sequence will show:
1. strong same-sequence performance on the source domain;
2. partial zero-shot transfer to related plain targets;
3. clear domain shift when transferred to enhanced targets;
4. measurable improvement with small-scale target-domain fine-tuning.

## Modules

### Module A — Cohort audit and normalization
Outputs:
- `manifest_reviewed.csv`
- sequence distribution table
- fold balance table
- optional vendor imbalance table
- audit warnings

### Module B — Main source-sequence baseline
Source sequence:
- `T2`

Model:
- `nnUNetv2 3d_fullres`

Split rule:
- strict patient-level 5-fold CV

Main results:
- Table 2
- Figure 3

### Module C — Zero-shot generalization
Train on:
- `T2`

Directly test on:
- `T2WI`, `DWI`, `ADC`, `C-pre`, `ARTERIAL`, `PORTAL`, `DELAY`

Main results:
- Table 3
- Figure 4

### Module D — Few-shot adaptation
Target recommendations:
- `PORTAL`
- `ARTERIAL`

Compare:
- zero-shot
- 10% target fine-tuning
- 20% target fine-tuning

Main results:
- Table 4
- Figure 5

## Publication-oriented endpoints
- per-case metrics
- per-fold metrics
- macro average across target groups
- lesion-level detection metrics
- failure-case examples
- objective self-audit

## Minimum publishable package
1. T2-only source 5-fold main baseline
2. T2→other-sequence zero-shot matrix
3. one important enhanced target few-shot adaptation experiment
4. self-audit and limitations
5. paper-ready manuscript draft
