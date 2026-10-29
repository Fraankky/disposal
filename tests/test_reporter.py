"""Reporter: header tetap walau nol event, baris per event, summary."""

from pathlib import Path

from disposal.area_rules import Event
from disposal.reporter import Reporter


def event(tid=1, bbox=(120.0, 110.0, 220.0, 190.0)) -> Event:
    return Event(
        track_id=tid,
        class_name="Truck",
        confidence=0.91,
        frame_index=7,
        video_time_seconds=0.280,
        bbox=bbox,
        coverage_ratio=1.0,
    )


def test_csv_header_only_when_no_event(tmp_path):
    r = Reporter(tmp_path, run_id="r1", video_name="v.mp4")
    r.finish("completed", 0, 0.1, {})
    lines = (tmp_path / "violations.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("run_id,event_id,video_name")


def test_event_row_and_id(tmp_path):
    r = Reporter(tmp_path, run_id="abc123", video_name="v.mp4")
    r.write_event(event(), "abc123-0001")
    r.finish("completed", 100, 4.0, {"VIOLATION": 1})
    text = (tmp_path / "violations.csv").read_text(encoding="utf-8")
    assert "abc123-0001,v.mp4,violation_area,1,Truck,0.9100,7,0.280" in text.replace("\n", ",")
    assert text.startswith("run_id")
    summary = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "completed" in summary and '"events": 1' in summary
