# Scripts Layout

This project uses these script namespaces:

- `common`
  Shared helpers.
- `dataset`
  Manifest, fold, and dataset export logic.
- `preprocess`
  Data preprocessing such as liver ROI generation.
- `experiment`
  Experiment generation and registration planning.
- `eval`
  Prediction evaluation and postprocessing.
- `qc`
  Quality-control utilities.
- `workflow`
  High-level workflow entry points.

Run scripts with:

```powershell
python -m scripts.<namespace>.<module> [args...]
```

Examples:

```powershell
python -m scripts.preprocess.run_totalsegmentator_liver_roi --help
python -m scripts.experiment.generate_sequence_screening_jobs --help
python -m scripts.workflow.bootstrap_workspace --help
```
