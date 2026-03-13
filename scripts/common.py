from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def materialize_file(src: str | Path, dst: str | Path, mode: str) -> str:
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"

    try:
        dst.symlink_to(src)
        return "symlink"
    except OSError:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def is_label_file(path: Path, label_keywords: Iterable[str]) -> bool:
    lower = path.name.lower()
    return any(keyword.lower() in lower for keyword in label_keywords)


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def extract_case_stem(path: str | Path) -> str:
    stem = strip_nii_suffix(Path(path).name)
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


def split_case_stem(case_stem: str) -> Tuple[str, str]:
    if "_" not in case_stem:
        return case_stem, case_stem
    patient_id, seq_raw = case_stem.rsplit("_", 1)
    return patient_id, seq_raw


def sanitize_case_id(case_stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", case_stem).strip("_")


def make_case_id(case_stem: str) -> str:
    return sanitize_case_id(case_stem)


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def get_canonical_sequences(cfg: dict) -> List[str]:
    return list(cfg.get("sequence_aliases", {}).keys())


def get_named_collection_map(cfg: dict, sections: Sequence[str] = ("sequence_bundles", "target_bundles")) -> Dict[str, List[str]]:
    collection_map: Dict[str, List[str]] = {}
    for section in sections:
        entries = cfg.get(section, {})
        for name, value in entries.items():
            if isinstance(value, dict):
                members = value.get("members") or value.get("sequences") or value.get("targets") or value.get("bundles")
            else:
                members = value
            if members is None:
                continue
            collection_map[name] = list(members)
    return collection_map


def expand_named_collection(
    name: str,
    cfg: dict,
    sections: Sequence[str] = ("sequence_bundles", "target_bundles"),
    stack: Optional[List[str]] = None,
) -> List[str]:
    canonical = set(get_canonical_sequences(cfg))
    if name in canonical:
        return [name]

    collection_map = get_named_collection_map(cfg, sections)
    if name not in collection_map:
        raise ValueError(f"Unknown collection: {name}. Available: {sorted(collection_map)}")

    stack = stack or []
    if name in stack:
        raise ValueError(f"Recursive collection definition detected: {' -> '.join(stack + [name])}")

    expanded: List[str] = []
    for item in collection_map[name]:
        if item in canonical:
            expanded.append(item)
        elif item in collection_map:
            expanded.extend(expand_named_collection(item, cfg, sections=sections, stack=stack + [name]))
        else:
            raise ValueError(f"Unknown collection member `{item}` while expanding `{name}`")
    return unique_preserve_order(expanded)


def resolve_experiment(cfg: dict, experiment_id: str) -> dict:
    experiments = cfg.get("experiments", {})
    if experiment_id not in experiments:
        raise ValueError(f"Unknown experiment_id: {experiment_id}. Available: {sorted(experiments)}")
    return experiments[experiment_id]


def resolve_label_for_image(image_path: Path, label_dir: Path) -> Optional[Path]:
    case_stem = extract_case_stem(image_path)
    candidates = [label_dir / f"{case_stem}.nii.gz", label_dir / f"{case_stem}.nii"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def detect_nnunet_subsets(root: str | Path) -> List[Tuple[str, Path, Optional[Path]]]:
    root = Path(root)
    subset_specs: Sequence[Tuple[str, str, str]] = (
        ("train", "imagesTr", "labelsTr"),
        ("test", "imagesTs", "labelsTs"),
    )
    found: List[Tuple[str, Path, Optional[Path]]] = []
    for subset_name, image_dir_name, label_dir_name in subset_specs:
        image_dir = root / image_dir_name
        label_dir = root / label_dir_name
        if image_dir.exists():
            found.append((subset_name, image_dir, label_dir if label_dir.exists() else None))
    return found


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
    stem = extract_case_stem(path)
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
    stem = extract_case_stem(path)
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
