from pathlib import Path

from app.config import resolve_default_singing_model


def test_resolve_default_singing_model_returns_score_when_artifact_exists(tmp_path: Path):
    model_dir = tmp_path / "data" / "models" / "singing_candidate"
    model_dir.mkdir(parents=True)
    (model_dir / "model.joblib").write_bytes(b"placeholder")

    model_path, mode = resolve_default_singing_model(tmp_path)

    assert model_path == model_dir.resolve()
    assert mode == "score"


def test_resolve_default_singing_model_returns_off_when_missing(tmp_path: Path):
    model_path, mode = resolve_default_singing_model(tmp_path)

    assert model_path is None
    assert mode == "off"
