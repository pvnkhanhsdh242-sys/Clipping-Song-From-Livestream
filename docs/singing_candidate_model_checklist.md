# Singing Candidate Model Implementation Checklist

## Status

- [x] Created continuation checklist.
- [x] Inspect current config, pipeline, manifest, preview, and test surfaces.
- [x] Add singing model feature extraction, label loading, scorer, and trainer.
- [x] Integrate singing scoring/filtering into CLI config and pipeline.
- [x] Extend manifest and preview outputs.
- [x] Update sample config and README training workflow docs.
- [x] Add unit and smoke tests.
- [x] Run verification.

## Changed Files

- `docs/singing_candidate_model_checklist.md`
- `app/config.py`
- `app/main.py`
- `app/output/manifest.py`
- `app/output/preview.py`
- `app/singing/`
- `app/ui/streamlit_app.py`
- `config.sample.yaml`
- `README.md`
- `requirements.txt`
- `scripts/build_singing_clip_manifest.py`
- `scripts/train_singing_candidate_model.py`
- `tests/test_main_split_segments.py`
- `tests/test_manifest.py`
- `tests/test_singing_features.py`
- `tests/test_singing_labels.py`
- `tests/test_singing_scorer.py`
- `tests/test_singing_training.py`

## Verification Commands

- Passed: `.venv\Scripts\python.exe -m pytest -q` (`35 passed, 1 skipped`)
- Passed: `.venv\Scripts\python.exe -m app.main --help` exposes singing model flags.
- Passed: `scripts\train_singing_candidate_model.py --manifest output\training_smoke\synthetic_manifest.csv --output-dir output\training_smoke\model_5_epochs --epochs 5 --validation-size 0.5`
- Passed: reloaded `output\training_smoke\model_5_epochs` and scored a synthetic singing window (`score=0.9602`).
- Passed: `scripts\build_singing_clip_manifest.py` generated `output\singing_clip_train_positive.csv` with 38 rows from active clip folders.
- Passed: trained positive-only smoke model at `output\singing_model_positive_only`; trainer correctly fell back to dummy classifier because there were 38 positive and 0 negative labels.

## Current Blockers

- None.

## Notes For Continuation

- V1 is candidate-level scoring only.
- Labels come from reviewed manifest CSV/JSON rows via `label_singing`.
- Runtime must remain backward-compatible when singing model mode is `off`.
- The first trainer writes `model.joblib` and `metadata.json` to `data/models/singing_candidate` by default.
- `scripts/train_singing_candidate_model.py --epochs` maps to the sklearn baseline solver iteration budget.
- Very short debug runs such as `--epochs 5` can emit sklearn convergence warnings; use a higher value for real reviewed manifests.
- Clip-folder training needs both `label_singing=1` and `label_singing=0`; positive-only clip folders only validate the loading path.
