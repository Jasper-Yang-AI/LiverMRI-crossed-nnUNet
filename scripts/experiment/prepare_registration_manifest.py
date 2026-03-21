from __future__ import annotations

import argparse
import json

import pandas as pd

from scripts.common.common import ensure_dir, expand_named_collection, load_yaml, make_case_id


def pick_anchor_row(group: pd.DataFrame, anchor_members: list[str]) -> pd.Series | None:
    for member in anchor_members:
        matches = group[group["seq_group"].astype(str) == str(member)].copy()
        if not matches.empty:
            return matches.sort_values(by=["subset", "case_stem"]).iloc[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare a patient-wise registration job manifest after sequence screening.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/dataset/livermri_crossseq_dataset.yaml")
    parser.add_argument("--anchor-target", default=None, help="Logical target/bundle, for example PORTAL or T2_MAIN.")
    parser.add_argument("--anchor-json", default=None, help="Optional recommended_anchor.json produced by summarize_sequence_screening.py")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    out_dir = ensure_dir(args.out_dir)

    anchor_target = args.anchor_target
    if not anchor_target and args.anchor_json:
        with open(args.anchor_json, "r", encoding="utf-8") as handle:
            anchor_target = json.load(handle).get("recommended_anchor_target")
    if not anchor_target:
        raise ValueError("Provide `--anchor-target` or `--anchor-json`.")

    anchor_members = expand_named_collection(anchor_target, cfg, sections=("target_bundles", "sequence_bundles"))
    manifest = pd.read_csv(args.manifest)
    pair_rows = []
    patient_rows = []

    for patient_id, group in manifest.groupby("patient_id"):
        group = group.copy()
        anchor_row = pick_anchor_row(group, anchor_members)
        if anchor_row is None:
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "anchor_target": anchor_target,
                    "status": "missing_anchor",
                    "n_available_sequences": int(group.shape[0]),
                }
            )
            continue

        anchor_case_id = make_case_id(str(anchor_row["case_stem"]))
        patient_rows.append(
            {
                "patient_id": patient_id,
                "anchor_target": anchor_target,
                "anchor_case_id": anchor_case_id,
                "anchor_seq_group": anchor_row["seq_group"],
                "anchor_seq_raw": anchor_row["seq_raw"],
                "status": "ok",
                "n_available_sequences": int(group.shape[0]),
            }
        )

        for _, moving_row in group.iterrows():
            moving_case_id = make_case_id(str(moving_row["case_stem"]))
            if moving_case_id == anchor_case_id:
                continue
            job_dir = out_dir / "jobs" / str(patient_id) / f"{moving_case_id}_to_{anchor_case_id}"
            pair_rows.append(
                {
                    "patient_id": patient_id,
                    "subset": anchor_row.get("subset", moving_row.get("subset", "")),
                    "fold": anchor_row.get("fold", moving_row.get("fold", -1)),
                    "cohort_role": anchor_row.get("cohort_role", moving_row.get("cohort_role", "")),
                    "anchor_target": anchor_target,
                    "fixed_case_id": anchor_case_id,
                    "fixed_case_stem": anchor_row["case_stem"],
                    "fixed_seq_group": anchor_row["seq_group"],
                    "fixed_seq_raw": anchor_row["seq_raw"],
                    "fixed_image_path": anchor_row["image_path"],
                    "fixed_label_path": anchor_row["label_path"],
                    "fixed_roi_mask_clean_path": anchor_row.get("roi_mask_clean_path", ""),
                    "fixed_roi_mask_dilated_path": anchor_row.get("roi_mask_dilated_path", ""),
                    "moving_case_id": moving_case_id,
                    "moving_case_stem": moving_row["case_stem"],
                    "moving_seq_group": moving_row["seq_group"],
                    "moving_seq_raw": moving_row["seq_raw"],
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

    pairwise_manifest_path = out_dir / "pairwise_registration_manifest.csv"
    patient_summary_path = out_dir / "registration_patient_summary.csv"
    pd.DataFrame(pair_rows).to_csv(pairwise_manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(patient_rows).to_csv(patient_summary_path, index=False, encoding="utf-8-sig")

    meta = {
        "anchor_target": anchor_target,
        "anchor_members": anchor_members,
        "n_pairwise_jobs": len(pair_rows),
        "n_patients": len(patient_rows),
    }
    with open(out_dir / "registration_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"Saved pairwise registration manifest to: {pairwise_manifest_path}")
    print(f"Saved patient summary to: {patient_summary_path}")


if __name__ == "__main__":
    main()

