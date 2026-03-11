from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from common import ensure_dir, load_yaml


def make_case_id(patient_id: str, seq_group: str, row_idx: int) -> str:
    safe_patient = str(patient_id).replace(" ", "_")
    safe_seq = str(seq_group).replace("+", "PLUS").replace("-", "")
    return f"{safe_patient}__{safe_seq}__{row_idx:05d}"


def main():
    parser = argparse.ArgumentParser(description="Export source sequence data into nnUNet_raw format.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--seq-group", required=True)
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    df = pd.read_csv(args.manifest)
    df = df[df["seq_group"] == args.seq_group].copy()

    dataset_name = f"Dataset{args.dataset_id:03d}_LiverTumor_{args.seq_group}"
    base = Path(args.nnunet_raw) / dataset_name
    imagesTr = ensure_dir(base / "imagesTr")
    labelsTr = ensure_dir(base / "labelsTr")

    exported_rows = []
    for row_idx, row in df.iterrows():
        case_id = make_case_id(row["patient_id"], row["seq_group"], row_idx)
        src_img = Path(row["image_path"])
        src_lab = Path(row["label_path"])

        dst_img = imagesTr / f"{case_id}_0000.nii.gz"
        dst_lab = labelsTr / f"{case_id}.nii.gz"

        if args.copy_mode == "copy":
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lab, dst_lab)
        else:
            if dst_img.exists():
                dst_img.unlink()
            if dst_lab.exists():
                dst_lab.unlink()
            dst_img.symlink_to(src_img)
            dst_lab.symlink_to(src_lab)

        exported_rows.append(
            {
                "case_id": case_id,
                "patient_id": row["patient_id"],
                "seq_group": row["seq_group"],
                "fold": int(row["fold"]),
                "image_path": str(src_img),
                "label_path": str(src_lab),
            }
        )

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": cfg["labels"],
        "numTraining": len(exported_rows),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": f"Source sequence export for {args.seq_group}",
    }

    with open(base / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    pd.DataFrame(exported_rows).to_csv(base / "case_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Exported {len(exported_rows)} cases to {base}")


if __name__ == "__main__":
    main()
