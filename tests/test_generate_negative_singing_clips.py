import csv
from pathlib import Path

from scripts.generate_negative_singing_clips import (
    Interval,
    NegativeSample,
    find_gaps,
    load_manifest_positive_intervals,
    merge_intervals,
    plan_negative_samples,
    resolve_local_vod_path,
    write_negative_manifest,
)


def test_merge_intervals_applies_padding_and_merges():
    merged = merge_intervals(
        [Interval(10.0, 20.0), Interval(25.0, 30.0)],
        pad_sec=3.0,
        duration_sec=100.0,
    )

    assert merged == [Interval(7.0, 33.0)]


def test_find_gaps_returns_minimum_length_gaps():
    gaps = find_gaps(
        [Interval(10.0, 20.0), Interval(50.0, 70.0)],
        duration_sec=100.0,
        min_gap_sec=25.0,
    )

    assert gaps == [Interval(20.0, 50.0), Interval(70.0, 100.0)]


def test_resolve_local_vod_path_repairs_container_path(tmp_path: Path):
    run_root = tmp_path / "output" / "Run A"
    manifest = run_root / "manifests" / "abc_manifest.csv"
    vod = run_root / "vods" / "sample.mp4"
    manifest.parent.mkdir(parents=True)
    vod.parent.mkdir(parents=True)
    vod.write_bytes(b"placeholder")

    resolved = resolve_local_vod_path("/app/output/Run A/vods/sample.mp4", manifest)

    assert resolved == vod.resolve()


def test_load_manifest_positive_intervals_uses_run_vod(tmp_path: Path):
    run_root = tmp_path / "output" / "Run A"
    manifest = run_root / "manifests" / "abc_manifest.csv"
    vod = run_root / "vods" / "sample.mp4"
    manifest.parent.mkdir(parents=True)
    vod.parent.mkdir(parents=True)
    vod.write_bytes(b"placeholder")
    manifest.write_text(
        "source_video,start_sec,end_sec\n/app/output/Run A/vods/sample.mp4,10,20\n",
        encoding="utf-8",
    )

    resolved, intervals = load_manifest_positive_intervals(manifest)

    assert resolved == vod.resolve()
    assert intervals == [Interval(10.0, 20.0)]


def test_plan_negative_samples_balances_from_gaps(monkeypatch, tmp_path: Path):
    vod = tmp_path / "vod.mp4"
    vod.write_bytes(b"placeholder")
    manifest = tmp_path / "run" / "manifests" / "abc_manifest.csv"
    manifest.parent.mkdir(parents=True)

    monkeypatch.setattr("scripts.generate_negative_singing_clips.probe_duration_sec", lambda path: 500.0)

    samples = plan_negative_samples(
        [(manifest, vod, [Interval(100.0, 200.0), Interval(250.0, 350.0)])],
        output_dir=tmp_path / "neg",
        positive_pad_sec=10.0,
        min_negative_sec=30.0,
        max_sample_duration_sec=None,
        max_negatives=None,
        seed=1,
    )

    assert len(samples) == 2
    assert all(sample.duration_sec >= 30.0 for sample in samples)
    assert all(sample.output_path.parent == tmp_path / "neg" for sample in samples)


def test_plan_negative_samples_caps_duration_for_smoke(monkeypatch, tmp_path: Path):
    vod = tmp_path / "vod.mp4"
    vod.write_bytes(b"placeholder")
    manifest = tmp_path / "run" / "manifests" / "abc_manifest.csv"
    manifest.parent.mkdir(parents=True)

    monkeypatch.setattr("scripts.generate_negative_singing_clips.probe_duration_sec", lambda path: 500.0)

    samples = plan_negative_samples(
        [(manifest, vod, [Interval(100.0, 220.0)])],
        output_dir=tmp_path / "neg",
        positive_pad_sec=10.0,
        min_negative_sec=30.0,
        max_sample_duration_sec=45.0,
        max_negatives=None,
        seed=1,
    )

    assert len(samples) == 1
    assert samples[0].duration_sec == 45.0


def test_write_negative_manifest(tmp_path: Path):
    vod = tmp_path / "vod.mp4"
    clip = tmp_path / "negative.mp4"
    manifest = tmp_path / "negative.csv"
    sample = NegativeSample(vod_path=vod, start_sec=10.0, end_sec=40.0, output_path=clip)

    count = write_negative_manifest([sample], manifest)

    assert count == 1
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    assert rows[0]["source_video"] == str(clip)
    assert rows[0]["label_singing"] == "0"
    assert rows[0]["boundary_method"] == "vod_gap_negative"
