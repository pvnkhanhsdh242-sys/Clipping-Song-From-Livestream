import csv
import json
from pathlib import Path

import pytest

from app.singing.labels import load_labeled_candidates, parse_label_singing


def test_parse_label_singing_accepts_common_values():
    assert parse_label_singing("yes") == 1
    assert parse_label_singing("not_singing") == 0
    assert parse_label_singing(True) == 1
    assert parse_label_singing("") is None


def test_load_labeled_candidates_from_csv_and_json(tmp_path: Path):
    source = tmp_path / "sample.wav"
    source.write_bytes(b"placeholder")

    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_video", "start_sec", "end_sec", "label_singing"])
        writer.writeheader()
        writer.writerow({"source_video": str(source), "start_sec": "0", "end_sec": "1", "label_singing": "yes"})
        writer.writerow({"source_video": str(source), "start_sec": "1", "end_sec": "2", "label_singing": ""})

    json_path = tmp_path / "manifest.json"
    json_path.write_text(
        json.dumps(
            [
                {"source_video": str(source), "start_sec": 2, "end_sec": 3, "label_singing": "no"},
            ]
        ),
        encoding="utf-8",
    )

    labeled = load_labeled_candidates([csv_path, json_path])

    assert len(labeled) == 2
    assert [item.label_singing for item in labeled] == [1, 0]
    assert labeled[0].source_video == source


def test_load_labeled_candidates_rejects_unknown_label(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_video,start_sec,end_sec,label_singing\nsample.wav,0,1,maybe\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported label_singing"):
        load_labeled_candidates([manifest])
