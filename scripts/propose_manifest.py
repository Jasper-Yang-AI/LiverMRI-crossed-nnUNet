from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import infer_patient_from_path, infer_sequence_from_path, is_label_file, list_nii_files, load_yaml, normalize_sequence


def main():
    parser = argparse.ArgumentParser(description="Best-effort manifest proposal from a raw MRI root.")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    aliases = cfg["sequence_aliases"]
    label_keywords = cfg["label_keywords"]

    files = list_nii_files(args.root)

    rows = []
    image_candidates = [p for p in files if not is_label_file(p, label_keywords)]
    label_candidates = [p for p in files if is_label_file(p, label_keywords)]

    label_index = {}
    for p in label_candidates:
        stem = p.name.replace(".nii.gz", "").replace(".nii", "")
        label_index[stem] = str(p)

    for img in image_candidates:
        stem = img.name.replace(".nii.gz", "").replace(".nii", "")
        patient_id = infer_patient_from_path(img)
        seq_raw = infer_sequence_from_path(img)
        seq_group = normalize_sequence(seq_raw, aliases)

        # Best-effort label pairing by shared prefix in same folder or exact stem
        guessed_label = None
        for cand_stem, cand_path in label_index.items():
            if stem in cand_stem or cand_stem in stem:
                guessed_label = cand_path
                break

        rows.append(
            {
                "patient_id": patient_id,
                "seq_raw": seq_raw,
                "seq_group": seq_group,
                "image_path": str(img),
                "label_path": guessed_label,
                "vendor": "",
                "field_strength": "",
                "needs_manual_review": 1,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved proposed manifest to: {out}")
    print("Review all rows before training.")


if __name__ == "__main__":
    main()
