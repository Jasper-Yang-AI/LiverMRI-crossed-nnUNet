from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

from livermri_crossseq.config import load_yaml_config
from livermri_crossseq.registration.confidence import (
    build_anchor_confidence,
    build_overlap_confidence,
    load_resampled_array,
)
from scripts.common.common import ensure_dir, make_case_id
from scripts.common.roi_utils import (
    affine_after_crop,
    build_nifti,
    compute_bbox_slices,
    load_float_image,
    load_label_like,
    load_mask_like,
    save_nifti,
)


def load_float_like(path: str | Path, ref_image: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != ref_image.shape or not np.allclose(image.affine, ref_image.affine, atol=1e-3):
        image = resample_from_to(image, ref_image, order=1)
    return np.asarray(image.get_fdata(), dtype=np.float32)


def crop_array(array: np.ndarray, crop_slices: tuple[slice, slice, slice]) -> np.ndarray:
    return np.asarray(array[crop_slices], dtype=np.float32)


def find_registered_row(patient_jobs: pd.DataFrame, sequence_name: str) -> pd.Series | None:
    matches = patient_jobs[patient_jobs["moving_seq_group"].astype(str) == str(sequence_name)]
    if matches.empty:
        return None
    return matches.sort_values(by=["moving_case_id"]).iloc[0]


def build_channel_layout(config: dict) -> list[dict[str, Any]]:
    model_cfg = config["model"]
    candidate_sequences = list(model_cfg["candidate_sequences"])
    include_confidence = bool(model_cfg.get("include_confidence_channels", True))
    include_liver_mask = bool(model_cfg.get("include_liver_mask_channel", True))

    channel_layout: list[dict[str, Any]] = []
    channel_index = 0
    for sequence_name in candidate_sequences:
        image_channels = [channel_index]
        channel_index += 1
        confidence_channels: list[int] = []
        if include_confidence:
            confidence_channels = [channel_index]
            channel_index += 1
        channel_layout.append(
            {
                "sequence_name": sequence_name,
                "image_channels": image_channels,
                "confidence_channels": confidence_channels,
                "is_anchor": sequence_name == model_cfg["anchor_target"],
            }
        )
    if include_liver_mask:
        liver_mask_channel = channel_index
    else:
        liver_mask_channel = None
    return channel_layout, liver_mask_channel


def export_case(
    patient_group: pd.DataFrame,
    patient_jobs: pd.DataFrame,
    anchor_target: str,
    channel_layout: list[dict[str, Any]],
    liver_mask_channel: int | None,
    crossseq_cfg: dict,
    raw_base: Path,
) -> dict | None:
    anchor_rows = patient_group[patient_group["seq_group"].astype(str) == str(anchor_target)].copy()
    if anchor_rows.empty:
        return None
    anchor_row = anchor_rows.sort_values(by=["subset", "case_stem"]).iloc[0]
    case_id = make_case_id(str(anchor_row["case_stem"]))

    anchor_arr, anchor_obj = load_float_image(anchor_row["image_path"])
    anchor_label = load_label_like(anchor_row["label_path"], anchor_obj)
    roi_column = crossseq_cfg["registration"].get("roi_column", "roi_mask_clean_path")
    anchor_roi = load_mask_like(anchor_row[roi_column], anchor_obj)
    crop_margin_mm = float(crossseq_cfg["model"]["crop_margin_mm"])
    crop_slices = compute_bbox_slices(anchor_roi, anchor_obj.header.get_zooms()[:3], crop_margin_mm)
    cropped_affine = affine_after_crop(anchor_obj.affine, crop_slices)

    channels: list[np.ndarray] = []
    sequence_presence: dict[str, int] = {}
    for branch_spec in channel_layout:
        seq_name = branch_spec["sequence_name"]
        is_anchor = bool(branch_spec["is_anchor"])
        if is_anchor:
            image_array = crop_array(anchor_arr, crop_slices)
            confidence_array = crop_array(build_anchor_confidence(anchor_roi), crop_slices)
            sequence_presence[seq_name] = 1
        else:
            reg_row = find_registered_row(patient_jobs, seq_name)
            if reg_row is None or not Path(str(reg_row["registered_image_path_expected"])).exists():
                image_array = np.zeros(tuple(s.stop - s.start for s in crop_slices), dtype=np.float32)
                confidence_array = np.zeros_like(image_array, dtype=np.float32)
                sequence_presence[seq_name] = 0
            else:
                image_array = crop_array(load_float_like(reg_row["registered_image_path_expected"], anchor_obj), crop_slices)
                confidence_path = Path(str(reg_row.get("registration_confidence_path_expected", "")))
                if confidence_path.exists():
                    confidence_full = load_resampled_array(confidence_path, anchor_obj, order=1)
                else:
                    moving_liver_path = Path(str(reg_row.get("registered_liver_mask_path_expected", "")))
                    moving_liver = load_mask_like(moving_liver_path, anchor_obj) if moving_liver_path.exists() else None
                    confidence_full = build_overlap_confidence(
                        anchor_roi,
                        moving_liver,
                        smooth_sigma=float(crossseq_cfg["registration"].get("liver_confidence_sigma", 1.0)),
                    )
                confidence_array = crop_array(confidence_full, crop_slices)
                sequence_presence[seq_name] = 1

        channels.append(image_array.astype(np.float32))
        if branch_spec["confidence_channels"]:
            channels.append(confidence_array.astype(np.float32))

    cropped_anchor_roi = crop_array(anchor_roi, crop_slices)
    if liver_mask_channel is not None:
        channels.append(cropped_anchor_roi.astype(np.float32))

    cropped_label = anchor_label[crop_slices].astype(np.uint8)
    channel_names: dict[str, str] = {}
    channel_index = 0
    for branch_spec in channel_layout:
        seq_name = branch_spec["sequence_name"]
        channel_names[str(channel_index)] = seq_name
        channel_index += 1
        if branch_spec["confidence_channels"]:
            channel_names[str(channel_index)] = "rescale_to_0_1"
            channel_index += 1
    if liver_mask_channel is not None:
        channel_names[str(channel_index)] = "rescale_to_0_1"

    for idx, channel in enumerate(channels):
        out_image = build_nifti(channel, anchor_obj, affine=cropped_affine, dtype=np.float32)
        save_nifti(out_image, raw_base / "imagesTr" / f"{case_id}_{idx:04d}.nii.gz")

    out_label = build_nifti(cropped_label, anchor_obj, affine=cropped_affine, dtype=np.uint8)
    save_nifti(out_label, raw_base / "labelsTr" / f"{case_id}.nii.gz")

    return {
        "case_id": case_id,
        "patient_id": anchor_row["patient_id"],
        "anchor_case_stem": anchor_row["case_stem"],
        "anchor_target": anchor_target,
        "present_sequences": json.dumps(sequence_presence, ensure_ascii=False),
        "n_present_sequences": int(sum(sequence_presence.values())),
        "bbox_start_zyx": ",".join(str(int(s.start)) for s in crop_slices),
        "bbox_stop_zyx": ",".join(str(int(s.stop)) for s in crop_slices),
        "channel_names": json.dumps(channel_names, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a registered multi-branch dataset in nnUNet raw format.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairwise-manifest", required=True)
    parser.add_argument("--study-config", default="configs/dataset/livermri_crossseq_dataset.yaml")
    parser.add_argument("--crossseq-config", default="configs/experiment/livermri_crossseq_fusion.yaml")
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--dataset-id", type=int, default=None)
    args = parser.parse_args()

    study_cfg = load_yaml_config(args.study_config)
    crossseq_cfg = load_yaml_config(args.crossseq_config)
    model_cfg = crossseq_cfg["model"]
    dataset_id = int(args.dataset_id or crossseq_cfg["export"]["dataset_id"])
    dataset_name_suffix = crossseq_cfg["export"]["dataset_name_suffix"]
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name_suffix}"
    raw_base = Path(args.nnunet_raw) / dataset_name
    ensure_dir(raw_base / "imagesTr")
    ensure_dir(raw_base / "labelsTr")

    manifest = pd.read_csv(args.manifest)
    pairwise_manifest = pd.read_csv(args.pairwise_manifest)
    channel_layout, liver_mask_channel = build_channel_layout(crossseq_cfg)
    anchor_target = model_cfg["anchor_target"]
    min_sequences = int(model_cfg["minimum_present_sequences"])

    exported_rows: list[dict] = []
    for patient_id, patient_group in manifest.groupby("patient_id"):
        patient_jobs = pairwise_manifest[pairwise_manifest["patient_id"].astype(str) == str(patient_id)].copy()
        exported = export_case(patient_group, patient_jobs, anchor_target, channel_layout, liver_mask_channel, crossseq_cfg, raw_base)
        if exported is None:
            continue
        if exported["n_present_sequences"] < min_sequences:
            continue
        exported_rows.append(exported)

    channel_names: dict[str, str] = {}
    idx = 0
    for branch_spec in channel_layout:
        channel_names[str(idx)] = branch_spec["sequence_name"]
        idx += 1
        if branch_spec["confidence_channels"]:
            channel_names[str(idx)] = "rescale_to_0_1"
            idx += 1
    if liver_mask_channel is not None:
        channel_names[str(idx)] = "rescale_to_0_1"

    dataset_json = {
        "channel_names": channel_names,
        "labels": study_cfg["labels"],
        "numTraining": len(exported_rows),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Registered liver MRI multi-branch cross-sequence export",
        "livermri_crossseq": {
            "anchor_target": anchor_target,
            "channel_layout": channel_layout,
            "liver_mask_channel": liver_mask_channel,
            "branch_stem_channels": int(model_cfg["branch_stem_channels"]),
            "dropout": model_cfg["dropout"],
            "loss": model_cfg["loss"],
            "drop_anchor_sequence": bool(model_cfg.get("drop_anchor_sequence", False)),
            "augmentation": model_cfg.get("augmentation", {}),
        },
    }
    with open(raw_base / "dataset.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2, ensure_ascii=False)

    if crossseq_cfg["export"].get("save_case_manifest", True):
        pd.DataFrame(exported_rows).to_csv(raw_base / "case_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"Exported {len(exported_rows)} cases to {raw_base}")
    print(f"Anchor target: {anchor_target}")
    print(f"Trainer: {model_cfg['trainer_name']}")


if __name__ == "__main__":
    main()
