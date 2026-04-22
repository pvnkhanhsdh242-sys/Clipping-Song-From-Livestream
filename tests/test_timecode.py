from app.utils.timecode import seconds_to_timecode, timecode_to_seconds


def test_seconds_to_timecode_roundtrip():
    value = 45296.789
    tc = seconds_to_timecode(value)
    assert tc == "12:34:56.789"
    assert abs(timecode_to_seconds(tc) - value) < 1e-6


def test_timecode_without_millis():
    assert timecode_to_seconds("00:00:07") == 7.0
