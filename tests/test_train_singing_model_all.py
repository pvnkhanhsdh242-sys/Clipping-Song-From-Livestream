from pathlib import Path

from scripts.train_singing_model_all import discover_positive_clip_dirs, main


def test_discover_positive_clip_dirs_uses_active_run_clips(tmp_path: Path):
    output_root = tmp_path / "output"
    active = output_root / "Run A" / "clips"
    empty = output_root / "Run B" / "clips"
    old = output_root / "Run C" / "clips_old"
    active.mkdir(parents=True)
    empty.mkdir(parents=True)
    old.mkdir(parents=True)
    (active / "clip.mp4").write_bytes(b"placeholder")
    (old / "old.mp4").write_bytes(b"placeholder")

    dirs = discover_positive_clip_dirs(output_root)

    assert dirs == [active.resolve()]


def test_orchestrator_dry_run_plans_without_training(tmp_path: Path):
    output_root = tmp_path / "output"
    positive_dir = output_root / "Run A" / "clips"
    negative_dir = tmp_path / "data" / "training_clips" / "not_singing"
    positive_dir.mkdir(parents=True)
    negative_dir.mkdir(parents=True)
    (positive_dir / "positive.mp4").write_bytes(b"placeholder")
    (negative_dir / "negative.mp4").write_bytes(b"placeholder")

    exit_code = main(
        [
            "--output-root",
            str(output_root),
            "--positive-manifest",
            str(tmp_path / "output" / "positive.csv"),
            "--negative-manifest",
            str(tmp_path / "output" / "negative.csv"),
            "--negative-dir",
            str(negative_dir),
            "--auto-negative-dir",
            str(negative_dir / "auto"),
            "--model-dir",
            str(tmp_path / "model"),
            "--backend",
            "pytorch",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--run-eval",
            "0",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "model").exists()
