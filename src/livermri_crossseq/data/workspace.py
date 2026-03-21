from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import KFold

from livermri_crossseq.config import load_yaml_config
from scripts.dataset.propose_manifest import build_rows_from_nnunet_layout, build_rows_with_generic_heuristics
from scripts.common.common import detect_nnunet_subsets, ensure_dir, load_yaml, make_case_id
from scripts.common.roi_utils import mm_tag, sanitize_subset_name


def build_manifest(dataset_root: Path, study_config_path: str | Path, out_csv: Path) -> pd.DataFrame:
    cfg = load_yaml(study_config_path)
    aliases = cfg["sequence_aliases"]
    label_keywords = cfg["label_keywords"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if detect_nnunet_subsets(dataset_root):
        df = build_rows_from_nnunet_layout(dataset_root, aliases, out_csv)
    else:
        df = build_rows_with_generic_heuristics(dataset_root, aliases, label_keywords, out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def assign_patient_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 3407) -> pd.DataFrame:
    result = df.copy()
    if "subset" not in result.columns:
        result["subset"] = "train"
    train_mask = result["subset"].astype(str).isin({"train", "imagesTr"})
    train_patients = sorted(result.loc[train_mask, "patient_id"].astype(str).unique().tolist())
    if len(train_patients) < n_folds:
        raise ValueError(f"Need at least {n_folds} training patients, found {len(train_patients)}")

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    patient_to_fold: dict[str, int] = {}
    for fold_id, (_, val_indices) in enumerate(splitter.split(train_patients)):
        for index in val_indices:
            patient_to_fold[train_patients[index]] = fold_id

    result["fold"] = -1
    result.loc[train_mask, "fold"] = result.loc[train_mask, "patient_id"].astype(str).map(patient_to_fold).astype(int)
    result["cohort_role"] = result["subset"].astype(str).apply(
        lambda subset: "internal_cv" if subset in {"train", "imagesTr"} else "external_test"
    )
    return result


def attach_roi_columns(df: pd.DataFrame, roi_dir: Path, dilation_mm: float) -> pd.DataFrame:
    roi_df = df.copy()
    dilation_dirname = f"dilated_{mm_tag(dilation_mm)}"

    def build_case_paths(row: pd.Series) -> pd.Series:
        case_id = make_case_id(str(row["case_stem"]))
        subset_name = sanitize_subset_name(row.get("subset", "unknown"))
        clean_path = roi_dir / "clean" / subset_name / f"{case_id}.nii.gz"
        dilated_path = roi_dir / dilation_dirname / subset_name / f"{case_id}.nii.gz"
        raw_path = roi_dir / "raw" / subset_name / f"{case_id}.nii.gz"
        return pd.Series(
            {
                "case_id": case_id,
                "roi_mask_raw_path": str(raw_path),
                "roi_mask_clean_path": str(clean_path),
                "roi_mask_dilated_path": str(dilated_path),
                "roi_available": int(clean_path.exists() and dilated_path.exists()),
            }
        )

    roi_cols = roi_df.apply(build_case_paths, axis=1)
    return pd.concat([roi_df.drop(columns=[c for c in roi_cols.columns if c in roi_df.columns], errors="ignore"), roi_cols], axis=1)


def create_workspace_dirs(workspace_root: Path) -> dict[str, Path]:
    dirs = {
        "root": workspace_root,
        "manifests": workspace_root / "manifests",
        "registration": workspace_root / "registration",
        "datasets": workspace_root / "datasets",
        "logs": workspace_root / "logs",
        "nnunet_raw": workspace_root / "nnUNet_raw",
        "nnunet_preprocessed": workspace_root / "nnUNet_preprocessed",
        "nnunet_results": workspace_root / "nnUNet_results",
        "exports": workspace_root / "exports",
    }
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def write_workspace_env(ps1_path: Path, workspace_dirs: dict[str, Path], extra_env: dict[str, str] | None = None) -> None:
    lines = [
        f'$env:nnUNet_raw = "{workspace_dirs["nnunet_raw"]}"',
        f'$env:nnUNet_preprocessed = "{workspace_dirs["nnunet_preprocessed"]}"',
        f'$env:nnUNet_results = "{workspace_dirs["nnunet_results"]}"',
        f'$env:LIVERMRI_CROSSSEQ_WORKSPACE = "{workspace_dirs["root"]}"',
    ]
    for key, value in (extra_env or {}).items():
        lines.append(f'$env:{key} = "{value}"')
    ps1_path.parent.mkdir(parents=True, exist_ok=True)
    ps1_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_manifest(df: pd.DataFrame, roi_enabled: bool) -> dict:
    summary = {
        "n_rows": int(df.shape[0]),
        "n_patients": int(df["patient_id"].nunique()),
        "subset_counts": df["subset"].fillna("UNKNOWN").value_counts().to_dict() if "subset" in df.columns else {},
        "sequence_counts": df["seq_group"].fillna("UNKNOWN").value_counts().to_dict() if "seq_group" in df.columns else {},
        "roi_enabled": bool(roi_enabled),
    }
    if "roi_available" in df.columns:
        summary["n_roi_available"] = int(df["roi_available"].sum())
    if "fold" in df.columns:
        summary["fold_counts"] = df[df["fold"] >= 0]["fold"].value_counts().sort_index().to_dict()
    return summary


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
