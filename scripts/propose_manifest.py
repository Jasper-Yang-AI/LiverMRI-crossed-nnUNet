from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    detect_nnunet_subsets,
    extract_case_stem,
    infer_patient_from_path,
    infer_sequence_from_path,
    is_label_file,
    list_nii_files,
    load_yaml,
    normalize_sequence,
    resolve_label_for_image,
    split_case_stem,
)


def build_rows_from_nnunet_layout(root: Path, aliases: dict, out_path: Path) -> pd.DataFrame:
    rows = []
    for subset_name, image_dir, label_dir in detect_nnunet_subsets(root):
        image_files = sorted(
            [
                p
                for p in image_dir.iterdir()
                if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
            ]
        )
        for image_path in image_files:
            case_stem = extract_case_stem(image_path)
            patient_id, seq_raw = split_case_stem(case_stem)
            label_path = resolve_label_for_image(image_path, label_dir) if label_dir else None
            seq_group = normalize_sequence(seq_raw, aliases)

            rows.append(
                {
                    "subset": subset_name,
                    "case_stem": case_stem,
                    "patient_id": patient_id,
                    "seq_raw": seq_raw,
                    "seq_group": seq_group,
                    "image_path": str(image_path),
                    "label_path": str(label_path) if label_path else "",
                    "label_available": int(label_path is not None),
                    "needs_manual_review": int(seq_group is None or label_path is None),
                    "notes": "",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No `.nii` / `.nii.gz` files found in nnUNet-style subsets under {root}")

    summary = {
        "layout": "nnunet_like",
        "n_rows": int(df.shape[0]),
        "n_patients": int(df["patient_id"].nunique()),
        "subset_counts": df["subset"].value_counts().to_dict(),
        "sequence_counts": df["seq_group"].fillna("UNKNOWN").value_counts().to_dict(),
        "n_missing_labels": int((df["label_available"] == 0).sum()),
        "n_unknown_sequences": int(df["seq_group"].isna().sum()),
    }
    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return df


def build_rows_with_generic_heuristics(root: Path, aliases: dict, label_keywords: list[str], out_path: Path) -> pd.DataFrame:
    files = list_nii_files(root)
    rows = []
    image_candidates = [p for p in files if not is_label_file(p, label_keywords)]
    label_candidates = [p for p in files if is_label_file(p, label_keywords)]

    label_index = {}
    for label_path in label_candidates:
        label_index[extract_case_stem(label_path)] = str(label_path)

    for image_path in image_candidates:
        case_stem = extract_case_stem(image_path)
        patient_id = infer_patient_from_path(image_path)
        seq_raw = infer_sequence_from_path(image_path)
        seq_group = normalize_sequence(seq_raw, aliases)

        guessed_label = label_index.get(case_stem, "")
        if not guessed_label:
            for cand_stem, cand_path in label_index.items():
                if case_stem in cand_stem or cand_stem in case_stem:
                    guessed_label = cand_path
                    break

        rows.append(
            {
                "subset": "unknown",
                "case_stem": case_stem,
                "patient_id": patient_id,
                "seq_raw": seq_raw,
                "seq_group": seq_group,
                "image_path": str(image_path),
                "label_path": guessed_label,
                "label_available": int(bool(guessed_label)),
                "needs_manual_review": 1,
                "notes": "heuristic_manifest",
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "layout": "generic_heuristic",
        "n_rows": int(df.shape[0]),
        "n_patients": int(df["patient_id"].nunique()) if not df.empty else 0,
        "sequence_counts": df["seq_group"].fillna("UNKNOWN").value_counts().to_dict() if not df.empty else {},
        "n_missing_labels": int((df["label_available"] == 0).sum()) if not df.empty else 0,
    }
    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build a reviewed manifest from a raw MRI root.")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.study_config)
    aliases = cfg["sequence_aliases"]
    label_keywords = cfg["label_keywords"]

    subset_defs = detect_nnunet_subsets(root)
    if subset_defs:
        df = build_rows_from_nnunet_layout(root, aliases, out)
        layout_name = "nnUNet-style"
    else:
        df = build_rows_with_generic_heuristics(root, aliases, label_keywords, out)
        layout_name = "generic heuristic"

    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved proposed manifest to: {out}")
    print(f"Detected layout: {layout_name}")
    print(f"Rows: {len(df)} | Patients: {df['patient_id'].nunique()}")
    print("Review rows with `needs_manual_review = 1` before training.")


if __name__ == "__main__":
    main()
