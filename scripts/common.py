from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_label_file(path: Path, label_keywords: Iterable[str]) -> bool:
    lower = path.name.lower()
    return any(keyword.lower() in lower for keyword in label_keywords)


def normalize_sequence(seq_raw: str, aliases: Dict[str, List[str]]) -> Optional[str]:
    if seq_raw is None:
        return None
    s = re.sub(r"[\s\-_/]+", "", seq_raw).upper()
    for canonical, alias_list in aliases.items():
        options = [canonical] + alias_list
        for alias in options:
            alias_norm = re.sub(r"[\s\-_/]+", "", alias).upper()
            if alias_norm == s:
                return canonical
    return None


def infer_patient_from_path(path: Path) -> str:
    # Best-effort heuristic:
    # 1) parent folder if not generic
    # 2) first token in filename split by "__" or "_"
    parent = path.parent.name
    if parent.lower() not in {"images", "labels", "image", "label"}:
        return parent
    stem = path.name.replace(".nii.gz", "").replace(".nii", "")
    if "__" in stem:
        return stem.split("__")[0]
    return stem.split("_")[0]


def infer_sequence_from_path(path: Path) -> str:
    # Best-effort heuristic:
    # 1) parent folder
    # 2) filename last token
    parent = path.parent.name
    if parent.lower() not in {"images", "labels", "image", "label"}:
        return parent
    stem = path.name.replace(".nii.gz", "").replace(".nii", "")
    if "__" in stem:
        return stem.split("__")[-1]
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else stem


def list_nii_files(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))])


def connected_components_3d(binary_array):
    from scipy import ndimage
    labeled, num = ndimage.label(binary_array)
    return labeled, num
