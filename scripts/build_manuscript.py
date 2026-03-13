from __future__ import annotations

import argparse
import json
from pathlib import Path

from jinja2 import Template


def main():
    parser = argparse.ArgumentParser(description="Build a manuscript draft from available study outputs.")
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    template_path = Path(args.template)
    out_path = Path(args.out)

    summary = {}
    summary_path = paper_dir / "study_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    template = Template(template_path.read_text(encoding="utf-8"))

    rendered = template.render(
        title="Cross-sequence generalization of nnUNetv2 for real-world heterogeneous liver MRI tumor segmentation: a patient-level retrospective study",
        abstract_background=(
            "Real-world liver MRI tumor segmentation is challenged by heterogeneous sequence composition across plain and enhanced acquisitions."
        ),
        abstract_purpose=(
            "To evaluate whether a dominant source-sequence nnUNetv2 model can generalize across heterogeneous target MRI sequences under strict patient-level partitioning."
        ),
        abstract_methods=(
            "A patient-level retrospective pipeline was established with sequence normalization, source-sequence 5-fold nnUNetv2 training, zero-shot cross-sequence evaluation, and optional few-shot adaptation."
        ),
        abstract_results=(
            f"A total of {summary.get('n_total_evals', 'N/A')} target evaluations were summarized, including {summary.get('n_internal_cv_evals', 'N/A')} internal CV evaluations and {summary.get('n_external_test_evals', 'N/A')} external test evaluations."
        ),
        abstract_conclusion=(
            "The source-sequence strategy provides a practical evaluation framework for studying transferability and domain shift in real-world heterogeneous liver MRI."
        ),
        introduction=(
            "Liver MRI in routine practice is inherently heterogeneous. Many patients undergo multiple plain sequences, and subsets receive enhanced sequences with distinct contrast characteristics. "
            "This heterogeneity complicates model deployment because segmentation networks trained on a single protocol may not transfer reliably across sequences. "
            "This study evaluates a pragmatic source-sequence-centered framework using nnUNetv2 under strict patient-level partitioning."
        ),
        methods=(
            "The study pipeline comprised cohort audit, sequence normalization, patient-level 5-fold construction, nnUNetv2 source-sequence training, zero-shot cross-sequence inference, optional few-shot adaptation, "
            "objective lesion-level evaluation, and an explicit self-audit of data leakage, sequence ambiguity, and confounding risks."
        ),
        results=(
            "Main internal same-sequence CV results, internal cross-sequence transfer matrices, external zero-shot transfer results, and adaptation gains should be inserted from the generated tables and figures in outputs/paper_assets."
        ),
        discussion=(
            "The discussion should interpret why source-to-similar plain transfer may outperform source-to-enhanced transfer, quantify domain shift, and acknowledge limitations such as retrospective design, potential vendor entanglement, and sequence-definition ambiguity."
        ),
        conclusion=(
            "A patient-level cross-sequence nnUNetv2 workflow can produce a publication-ready analysis of transferability and deployment risks in heterogeneous liver MRI tumor segmentation."
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Saved manuscript draft to: {out_path}")


if __name__ == "__main__":
    main()
