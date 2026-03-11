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
    parser = argparse.ArgumentParser(description="Export target sequence test sets for zero-shot inference.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/study_config.yaml")
    parser.add_argument("--source-seq", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    df = pd.read_csv(args.manifest)
    if "fold" not in df.columns:
        raise ValueError("Manifest must contain `fold` column.")

    out_dir = ensure_dir(args.out_dir)
    commands_ps = []
    commands_sh = []

    for target in args.targets:
        sub = df[df["seq_group"] == target].copy()
        if sub.empty:
            print(f"[WARN] No cases found for target: {target}")
            continue

        target_dir = ensure_dir(out_dir / target)
        imagesTs = ensure_dir(target_dir / "imagesTs")
        labelsTs = ensure_dir(target_dir / "labelsTs")
        rows = []

        for row_idx, row in sub.iterrows():
            case_id = make_case_id(row["patient_id"], row["seq_group"], row_idx)
            src_img = Path(row["image_path"])
            src_lab = Path(row["label_path"])
            dst_img = imagesTs / f"{case_id}_0000.nii.gz"
            dst_lab = labelsTs / f"{case_id}.nii.gz"

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

            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": row["patient_id"],
                    "seq_group": row["seq_group"],
                    "fold": int(row["fold"]),
                    "image_path": str(src_img),
                    "label_path": str(src_lab),
                }
            )

        pd.DataFrame(rows).to_csv(target_dir / "target_manifest.csv", index=False, encoding="utf-8-sig")

        pred_dir = out_dir / ".." / "predictions" / f"{args.source_seq}_to_{target}"
        pred_dir = pred_dir.resolve()

        ps_cmd = (
            f'nnUNetv2_predict -i "{imagesTs}" -o "{pred_dir}" '
            f'-d 301 -c {cfg["nnunet"]["configuration"]} -f 0 1 2 3 4'
        )
        sh_cmd = ps_cmd
        commands_ps.append(ps_cmd)
        commands_sh.append(sh_cmd)

    commands_dir = ensure_dir(out_dir.parent / "commands")
    (commands_dir / "infer_targets.ps1").write_text("\n".join(commands_ps), encoding="utf-8")
    (commands_dir / "infer_targets.sh").write_text("\n".join(commands_sh), encoding="utf-8")

    meta = {"source_seq": args.source_seq, "targets": args.targets}
    with open(out_dir.parent / "targets_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved target test sets to {out_dir}")
    print(f"Saved inference command files to {commands_dir}")


if __name__ == "__main__":
    main()
