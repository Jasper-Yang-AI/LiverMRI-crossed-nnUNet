from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from livermri_crossseq.config import load_yaml_config
from scripts.common.common import ensure_dir, expand_named_collection, make_case_id


def normalize_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = series.fillna(series.min() if not higher_is_better else series.max())
    if values.nunique() <= 1:
        return pd.Series(np.ones(len(values)), index=series.index)
    scaled = (values - values.min()) / (values.max() - values.min())
    return scaled if higher_is_better else 1.0 - scaled


def pick_anchor_row(group: pd.DataFrame, anchor_members: list[str]) -> pd.Series | None:
    for member in anchor_members:
        matches = group[group["seq_group"].astype(str) == str(member)].copy()
        if not matches.empty:
            return matches.sort_values(by=["subset", "case_stem"]).iloc[0]
    return None


def build_pilot_jobs(
    manifest: pd.DataFrame,
    patient_ids: list[str],
    anchor_target: str,
    study_cfg: dict,
    out_dir: Path,
) -> Path:
    anchor_members = expand_named_collection(anchor_target, study_cfg, sections=("target_bundles", "sequence_bundles"))
    pair_rows: list[dict] = []
    for patient_id in patient_ids:
        group = manifest[manifest["patient_id"].astype(str) == str(patient_id)].copy()
        anchor_row = pick_anchor_row(group, anchor_members)
        if anchor_row is None:
            continue
        anchor_case_id = make_case_id(str(anchor_row["case_stem"]))
        for _, moving_row in group.iterrows():
            moving_case_id = make_case_id(str(moving_row["case_stem"]))
            if moving_case_id == anchor_case_id:
                continue
            job_dir = out_dir / anchor_target / str(patient_id) / f"{moving_case_id}_to_{anchor_case_id}"
            pair_rows.append(
                {
                    "patient_id": patient_id,
                    "anchor_target": anchor_target,
                    "fixed_case_id": anchor_case_id,
                    "fixed_seq_group": anchor_row["seq_group"],
                    "fixed_image_path": anchor_row["image_path"],
                    "fixed_label_path": anchor_row["label_path"],
                    "fixed_roi_mask_clean_path": anchor_row.get("roi_mask_clean_path", ""),
                    "fixed_roi_mask_dilated_path": anchor_row.get("roi_mask_dilated_path", ""),
                    "moving_case_id": moving_case_id,
                    "moving_seq_group": moving_row["seq_group"],
                    "moving_image_path": moving_row["image_path"],
                    "moving_label_path": moving_row["label_path"],
                    "moving_roi_mask_clean_path": moving_row.get("roi_mask_clean_path", ""),
                    "moving_roi_mask_dilated_path": moving_row.get("roi_mask_dilated_path", ""),
                    "job_dir": str(job_dir),
                    "registered_image_path_expected": str(job_dir / "registered_image.nii.gz"),
                    "registered_liver_mask_path_expected": str(job_dir / "registered_liver_mask.nii.gz"),
                    "registered_label_path_expected": str(job_dir / "registered_label.nii.gz"),
                    "registration_confidence_path_expected": str(job_dir / "registration_confidence.nii.gz"),
                    "jacobian_determinant_path_expected": str(job_dir / "jacobian_determinant.nii.gz"),
                    "registration_metrics_path_expected": str(job_dir / "registration_metrics.json"),
                    "transform_path_expected": str(job_dir / "transform.tfm"),
                }
            )
    out_path = out_dir / anchor_target / "pilot_pairwise_registration_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan anchor shortlist benchmarking and pilot registration jobs.")
    parser.add_argument("--screening-summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/dataset/livermri_crossseq_dataset.yaml")
    parser.add_argument("--crossseq-config", default="configs/experiment/livermri_crossseq_fusion.yaml")
    parser.add_argument("--out-dir", default="workspaces/livermri_crossseq/registration/anchor_pilot")
    args = parser.parse_args()

    study_cfg = load_yaml_config(args.study_config)
    crossseq_cfg = load_yaml_config(args.crossseq_config)
    out_dir = ensure_dir(args.out_dir)

    summary = pd.read_csv(args.screening_summary)
    manifest = pd.read_csv(args.manifest)

    coverage = (
        manifest.groupby("seq_group")["patient_id"]
        .nunique()
        .reset_index()
        .rename(columns={"patient_id": "n_patients_available"})
    )
    merged = summary.merge(coverage, left_on="primary_target", right_on="seq_group", how="left")
    weights = crossseq_cfg["anchor"]["selection_weights"]
    merged["score_external_dice"] = normalize_score(merged.get("external_dice_mean", pd.Series(dtype=float)), True)
    merged["score_lesion_f1"] = normalize_score(merged.get("external_lesion_f1_mean", pd.Series(dtype=float)), True)
    merged["score_coverage"] = normalize_score(merged.get("n_patients_available", pd.Series(dtype=float)), True)
    merged["score_external_hd95"] = normalize_score(merged.get("external_hd95_mean", pd.Series(dtype=float)), False)
    merged["anchor_score"] = (
        weights["external_dice"] * merged["score_external_dice"]
        + weights["lesion_f1"] * merged["score_lesion_f1"]
        + weights["coverage"] * merged["score_coverage"]
        + weights["external_hd95"] * merged["score_external_hd95"]
    )
    merged = merged.sort_values("anchor_score", ascending=False).reset_index(drop=True)

    shortlist = crossseq_cfg["anchor"].get("shortlist") or merged["primary_target"].head(3).tolist()
    manifest_seq_sets = manifest.groupby("patient_id")["seq_group"].apply(lambda s: set(s.astype(str).tolist()))
    eligible_patients = [
        patient_id for patient_id, seqs in manifest_seq_sets.items() if all(candidate in seqs for candidate in shortlist)
    ]
    eligible_patients = sorted(map(str, eligible_patients))

    pilot_cases = int(crossseq_cfg["anchor"]["pilot_cases"])
    pilot_patients = eligible_patients[:pilot_cases]

    merged.to_csv(out_dir / "anchor_candidate_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"patient_id": pilot_patients}).to_csv(out_dir / "pilot_patients.csv", index=False, encoding="utf-8-sig")

    pilot_manifests: dict[str, str] = {}
    for anchor_target in shortlist:
        pilot_manifest = build_pilot_jobs(manifest, pilot_patients, anchor_target, study_cfg, out_dir)
        pilot_manifests[anchor_target] = str(pilot_manifest)

    recommendation = {
        "shortlist": shortlist,
        "recommended_anchor": shortlist[0] if shortlist else "",
        "n_eligible_patients": len(eligible_patients),
        "n_pilot_patients": len(pilot_patients),
        "pilot_manifests": pilot_manifests,
    }
    with open(out_dir / "pilot_recommendation.json", "w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, indent=2, ensure_ascii=False)

    print(f"Saved anchor table to: {out_dir / 'anchor_candidate_table.csv'}")
    print(f"Saved pilot patients to: {out_dir / 'pilot_patients.csv'}")
    print(f"Recommended shortlist: {shortlist}")


if __name__ == "__main__":
    main()
