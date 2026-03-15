from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import os

import pandas as pd

from scripts.lib.common import ensure_dir, load_yaml, make_case_id
from scripts.lib.roi_utils import (
    build_nifti,
    dilate_mask_mm,
    fill_mask_holes,
    largest_connected_component,
    load_float_image,
    load_mask_like,
    mm_tag,
    save_nifti,
    sanitize_subset_name,
)


def resolve_conda_executable() -> str:
    """Resolve a real conda executable path usable by subprocess on Windows."""
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    which_conda = shutil.which("conda")
    if which_conda:
        return which_conda

    for candidate in (
        Path.home() / "AppData" / "Local" / "anaconda3" / "condabin" / "conda.bat",
        Path.home() / "AppData" / "Local" / "anaconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "condabin" / "conda.bat",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
    ):
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Conda executable not found. Set CONDA_EXE, add conda to PATH, or pass --conda-env '' to run without conda."
    )


def build_command(args, image_path: Path, temp_dir: Path) -> list[str]:
    command: list[str] = []
    if args.conda_env:
        command.extend([resolve_conda_executable(), "run", "-n", args.conda_env])
    command.extend(
        [
            args.totalsegmentator_exe,
            "-i",
            str(image_path),
            "-o",
            str(temp_dir),
            "--task",
            args.task,
            "--roi_subset",
            args.roi_name,
        ]
    )
    if args.device:
        command.extend(["--device", args.device])
    if args.body_seg:
        command.append("--body_seg")
    return command


def locate_roi_output(temp_dir: Path, roi_name: str) -> Path:
    matches = sorted(temp_dir.rglob(f"{roi_name}.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"Could not find {roi_name}.nii.gz under {temp_dir}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Run TotalSegmentator liver ROI extraction for all cases in a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--study-config", default="configs/roi_study_config.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--conda-env", default=None, help="Optional conda env used via `conda run -n <env>`.")
    parser.add_argument("--totalsegmentator-exe", default="TotalSegmentator")
    parser.add_argument("--device", default=None, help="TotalSegmentator device, e.g. cpu, gpu, gpu:1, mps")
    parser.add_argument("--task", default=None)
    parser.add_argument("--roi-name", default=None)
    parser.add_argument("--dilation-mm", type=float, default=None)
    parser.add_argument("--min-liver-voxels", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--body-seg", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.study_config)
    roi_cfg = cfg.get("roi", {})
    args.task = args.task or roi_cfg.get("task", "total_mr")
    args.roi_name = args.roi_name or roi_cfg.get("roi_name", "liver")
    args.conda_env = args.conda_env or roi_cfg.get("conda_env")
    args.device = args.device or roi_cfg.get("device")
    args.dilation_mm = float(args.dilation_mm if args.dilation_mm is not None else roi_cfg.get("dilation_mm", 12.0))
    args.min_liver_voxels = int(
        args.min_liver_voxels if args.min_liver_voxels is not None else roi_cfg.get("min_liver_voxels", 5000)
    )

    dilation_dirname = f"dilated_{mm_tag(args.dilation_mm)}"
    out_dir = ensure_dir(args.out_dir)
    raw_root = ensure_dir(out_dir / "raw")
    clean_root = ensure_dir(out_dir / "clean")
    dilated_root = ensure_dir(out_dir / dilation_dirname)
    temp_root = ensure_dir(out_dir / "_tmp")

    manifest = pd.read_csv(args.manifest)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    # Fail fast for missing launchers so we don't emit thousands of identical per-row failures.
    if args.conda_env:
        _ = resolve_conda_executable()
    else:
        if not shutil.which(args.totalsegmentator_exe):
            raise FileNotFoundError(
                f"TotalSegmentator executable '{args.totalsegmentator_exe}' not found in PATH."
            )

    rows = []

    for _, row in manifest.iterrows():
        case_stem = str(row["case_stem"])
        case_id = make_case_id(case_stem)
        subset_name = sanitize_subset_name(row.get("subset", "unknown"))
        image_path = Path(row["image_path"])
        raw_mask_path = raw_root / subset_name / f"{case_id}.nii.gz"
        clean_mask_path = clean_root / subset_name / f"{case_id}.nii.gz"
        dilated_mask_path = dilated_root / subset_name / f"{case_id}.nii.gz"
        temp_dir = temp_root / subset_name / case_id

        base_row = {
            "case_id": case_id,
            "case_stem": case_stem,
            "patient_id": row.get("patient_id", ""),
            "seq_raw": row.get("seq_raw", ""),
            "seq_group": row.get("seq_group", ""),
            "subset": row.get("subset", ""),
            "image_path": str(image_path),
            "raw_mask_path": str(raw_mask_path),
            "clean_mask_path": str(clean_mask_path),
            "dilated_mask_path": str(dilated_mask_path),
            "dilation_mm": args.dilation_mm,
            "task": args.task,
            "roi_name": args.roi_name,
        }

        if args.skip_existing and raw_mask_path.exists() and clean_mask_path.exists() and dilated_mask_path.exists():
            rows.append({**base_row, "status": "skipped_existing"})
            continue

        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            ensure_dir(temp_dir)
            command = build_command(args, image_path, temp_dir)
            subprocess.run(command, check=True)

            raw_output = locate_roi_output(temp_dir, args.roi_name)
            _, image_obj = load_float_image(image_path)
            raw_mask = load_mask_like(raw_output, image_obj)
            clean_mask = raw_mask.copy()
            if roi_cfg.get("clean_largest_component", True):
                clean_mask = largest_connected_component(clean_mask)
            if roi_cfg.get("fill_holes", True):
                clean_mask = fill_mask_holes(clean_mask)
            spacing = image_obj.header.get_zooms()[:3]
            dilated_mask = dilate_mask_mm(clean_mask, spacing=spacing, radius_mm=args.dilation_mm)

            save_nifti(build_nifti(raw_mask, image_obj, dtype="uint8"), raw_mask_path)
            save_nifti(build_nifti(clean_mask, image_obj, dtype="uint8"), clean_mask_path)
            save_nifti(build_nifti(dilated_mask, image_obj, dtype="uint8"), dilated_mask_path)

            rows.append(
                {
                    **base_row,
                    "status": "ok",
                    "raw_voxels": int(raw_mask.sum()),
                    "clean_voxels": int(clean_mask.sum()),
                    "dilated_voxels": int(dilated_mask.sum()),
                    "below_min_liver_voxels": int(clean_mask.sum() < args.min_liver_voxels),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({**base_row, "status": f"failed: {exc}"})
            if args.fail_fast:
                raise
        finally:
            if not args.keep_temp and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "roi_generation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved ROI generation summary to: {summary_path}")
    if not summary.empty:
        print(summary["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

