from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from common import connected_components_3d


def load_binary(path: Path) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata()
    return (arr > 0).astype(np.uint8)


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    inter = (gt * pred).sum()
    denom = gt.sum() + pred.sum()
    return 1.0 if denom == 0 else float((2.0 * inter) / denom)


def surface_points(binary: np.ndarray) -> np.ndarray:
    if binary.sum() == 0:
        return np.zeros((0, 3))
    eroded = ndimage.binary_erosion(binary, iterations=1, border_value=0)
    surface = binary ^ eroded
    return np.argwhere(surface > 0)


def hd95(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_pts = surface_points(gt)
    pred_pts = surface_points(pred)
    if gt_pts.shape[0] == 0 and pred_pts.shape[0] == 0:
        return 0.0
    if gt_pts.shape[0] == 0 or pred_pts.shape[0] == 0:
        return float("inf")

    # Approximate HD95 with pairwise nearest distances
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)
    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(distances, 95))


def lesion_detection_metrics(gt: np.ndarray, pred: np.ndarray):
    gt_cc, gt_n = connected_components_3d(gt)
    pred_cc, pred_n = connected_components_3d(pred)

    matched_gt = set()
    matched_pred = set()

    for gi in range(1, gt_n + 1):
        gmask = gt_cc == gi
        for pi in range(1, pred_n + 1):
            pmask = pred_cc == pi
            if np.logical_and(gmask, pmask).any():
                matched_gt.add(gi)
                matched_pred.add(pi)

    recall = 1.0 if gt_n == 0 else len(matched_gt) / gt_n
    precision = 1.0 if pred_n == 0 else len(matched_pred) / pred_n
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return float(recall), float(precision), float(f1)


def volume_relative_error(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt.sum()
    p = pred.sum()
    return 0.0 if g == 0 else float(abs(p - g) / g)


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction folders against target labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    pred_root = Path(args.pred_root)

    rows = []
    for pred_dir in sorted(pred_root.glob("*")):
        if not pred_dir.is_dir():
            continue
        if "_to_" not in pred_dir.name:
            continue

        source_seq, target_seq = pred_dir.name.split("_to_", 1)
        target_rows = manifest[manifest["seq_group"] == target_seq].copy()

        for row_idx, row in tqdm(target_rows.iterrows(), total=target_rows.shape[0], desc=pred_dir.name):
            patient_id = row["patient_id"]
            safe_patient = str(patient_id).replace(" ", "_")
            safe_seq = str(target_seq).replace("+", "PLUS").replace("-", "")
            case_id = f"{safe_patient}__{safe_seq}__{row_idx:05d}"

            pred_path = pred_dir / f"{case_id}.nii.gz"
            gt_path = Path(row["label_path"])
            if not pred_path.exists() or not gt_path.exists():
                rows.append(
                    {
                        "source_seq": source_seq,
                        "target_seq": target_seq,
                        "patient_id": patient_id,
                        "fold": row.get("fold", -1),
                        "case_id": case_id,
                        "missing_prediction": 1,
                    }
                )
                continue

            gt = load_binary(gt_path)
            pred = load_binary(pred_path)
            rec, prec, f1 = lesion_detection_metrics(gt, pred)

            rows.append(
                {
                    "source_seq": source_seq,
                    "target_seq": target_seq,
                    "patient_id": patient_id,
                    "fold": row.get("fold", -1),
                    "case_id": case_id,
                    "missing_prediction": 0,
                    "dice": dice_score(gt, pred),
                    "hd95": hd95(gt, pred),
                    "lesion_recall": rec,
                    "lesion_precision": prec,
                    "lesion_f1": f1,
                    "volume_relative_error": volume_relative_error(gt, pred),
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {out_csv}")


if __name__ == "__main__":
    main()
