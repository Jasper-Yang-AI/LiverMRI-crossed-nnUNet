# Objective Self-Audit Framework

This study should be judged against the following risks:

## A. Data leakage
- Same patient must never appear in different folds.
- Same patient’s different sequences must never leak across train/test.

## B. Sequence-definition ambiguity
- `C+A/AP`, `C+V/VP`, `C+Delay/HP`, `C-pre/T1` mappings must be explicitly documented.
- Ambiguous rows must be resolved before training.

## C. Source-target imbalance
- Too few cases in a target sequence weaken conclusions.
- Macro average should be reported alongside raw target-specific metrics.

## D. Vendor confounding
- If vendor metadata are available, report vendor distribution by sequence.
- If sequence and vendor are strongly entangled, state this as a limitation.

## E. Label consistency
- Tumor label values must be consistent.
- Empty or broken labels must be flagged.

## F. Overclaiming
Do **not** claim:
- “first ever heterogeneous liver MRI segmentation study”
- “novel network architecture”
- “vendor-independent model” without stratified evidence

Prefer:
- “patient-level cross-sequence evaluation framework”
- “real-world heterogeneous routine MRI”
- “objective analysis of transferability and domain shift”
