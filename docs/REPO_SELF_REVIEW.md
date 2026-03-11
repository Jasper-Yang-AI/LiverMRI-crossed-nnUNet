# Repository Self Review

## What is complete
- A new local repository scaffold has been created.
- The study is locked to a paper-oriented workflow rather than an open-ended model-tweaking workflow.
- All critical publication steps are covered:
  - manifest proposal and review
  - strict patient-level fold assignment
  - source-sequence nnUNetv2 export
  - zero-shot target export
  - evaluation metrics
  - aggregation into paper tables
  - figure generation
  - manuscript drafting
  - self-audit

## What is objectively strong
1. The experiment is centered on a clean narrative: source-sequence training → cross-sequence zero-shot transfer → few-shot adaptation.
2. Leakage risk is explicitly handled at the patient level.
3. Sequence alias normalization is configurable.
4. Output assets are designed to map directly onto paper sections.

## What remains a limitation
1. Actual training is not executed in this environment because the real dataset and GPU runtime are not available here.
2. Manifest proposal is heuristic and must be manually reviewed before any claim is made.
3. Vendor analysis is optional and depends on available metadata.
4. The pipeline assumes tumor labels are binary by default; multiclass labels require config edits.
5. The generated manuscript is a structured draft, not a final polished submission.

## Bottom-line judgment
This repository is **publication-oriented and runnable on the user’s local machine**, but **not a substitute for real data curation and actual model execution**.
