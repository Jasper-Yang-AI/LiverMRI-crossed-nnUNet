# Suggested Paper Structure

## Title
Cross-sequence generalization of nnUNetv2 for real-world heterogeneous liver MRI tumor segmentation: a patient-level retrospective study

## Abstract
- Background
- Purpose
- Materials and methods
- Results
- Conclusion

## Introduction
1. Real-world liver MRI is heterogeneous across sequences.
2. Liver tumor segmentation models are often evaluated within single protocols.
3. Cross-sequence transferability in patient-level real-world cohorts remains insufficiently characterized.
4. Study aim and hypotheses.

## Materials and Methods
1. Cohort and inclusion
2. Sequence normalization and grouping
3. Patient-level fold construction
4. nnUNetv2 source-sequence training
5. Zero-shot target-sequence evaluation
6. Few-shot adaptation
7. Metrics
8. Statistical analysis
9. Self-audit framework

## Results
1. Cohort composition
2. Main source-sequence 5-fold results
3. Zero-shot cross-sequence results
4. Few-shot adaptation gains
5. Failure cases
6. Self-audit findings

## Discussion
1. Main findings
2. Why plain→plain transfers better than plain→enhanced
3. Practical deployment implications
4. Limitations
5. Future work

## Conclusion
The dominant source-sequence strategy provides a pragmatic workflow for publication-ready cross-sequence evaluation in real-world liver MRI tumor segmentation.
