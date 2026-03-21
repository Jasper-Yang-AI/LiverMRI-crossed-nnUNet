from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.common.roi_utils import build_nifti, ensure_binary_prediction, load_mask_like, save_nifti


def main():
    parser = argparse.ArgumentParser(description="Remove extra-hepatic predictions using a liver ROI mask.")
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--roi-column", default="roi_mask_dilated_path")
    parser.add_argument("--keep-original-when-missing-roi", action="store_true")
    args = parser.parse_args()

    eval_manifest = pd.read_csv(args.evaluation_manifest)
    out_root = Path(args.out_root)
    out_rows = []

    for row in eval_manifest.to_dict(orient="records"):
        prediction_path = Path(row["prediction_path"])
        roi_value = str(row.get(args.roi_column, "")).strip()
        roi_path = Path(roi_value) if roi_value else None
        dst_dir = out_root / str(row.get("eval_mode", "unknown")) / f"{row['source_seq']}_to_{row['target_seq']}"
        if int(row.get("fold", -1)) >= 0:
            dst_dir = dst_dir / f"fold_{int(row['fold'])}"
        dst_path = dst_dir / prediction_path.name

        updated_row = dict(row)
        updated_row["prediction_path_raw"] = str(prediction_path)
        updated_row["prediction_path_postprocessed"] = ""
        updated_row["postprocess_status"] = ""

        if not prediction_path.exists():
            updated_row["postprocess_status"] = "missing_prediction"
            out_rows.append(updated_row)
            continue

        if roi_path is None or not roi_path.exists():
            if args.keep_original_when_missing_roi:
                updated_row["prediction_path"] = str(prediction_path)
                updated_row["postprocess_status"] = "missing_roi_kept_original"
                out_rows.append(updated_row)
                continue
            updated_row["postprocess_status"] = "missing_roi"
            out_rows.append(updated_row)
            continue

        pred_arr, pred_img = ensure_binary_prediction(prediction_path)
        roi_mask = load_mask_like(roi_path, pred_img)
        filtered = (pred_arr * roi_mask).astype("uint8")
        save_nifti(build_nifti(filtered, pred_img, dtype="uint8"), dst_path)

        updated_row["prediction_path"] = str(dst_path)
        updated_row["prediction_path_postprocessed"] = str(dst_path)
        updated_row["postprocess_status"] = "ok"
        out_rows.append(updated_row)

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_manifest, index=False, encoding="utf-8-sig")
    print(f"Saved postprocessed evaluation manifest to: {out_manifest}")


if __name__ == "__main__":
    main()

