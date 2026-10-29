import csv
import json
from pathlib import Path

from .area_rules import REASON_FULL_INSIDE, VIOLATION, Event

CSV_FIELDS = [
    "run_id", "event_id", "video_name", "area_id", "track_id", "class_name", "confidence",
    "frame_index", "video_time_seconds", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "coverage_ratio", "status", "reason",
]


class Reporter:
    def __init__(self, run_dir: Path, *, run_id: str, video_name: str, area_id: str = "violation_area"):
        self.run_id = run_id
        self.video_name = video_name
        self.area_id = area_id
        self.run_dir = run_dir
        self.events_written = 0
        self._fh = (run_dir / "violations.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()  # CSV berheader tetap ada walau nol event

    def write_event(self, event: Event, event_id: str) -> None:
        row = {
            "run_id": self.run_id,
            "event_id": event_id,
            "video_name": self.video_name,
            "area_id": self.area_id,
            "track_id": event.track_id,
            "class_name": event.class_name or "",
            "confidence": f"{event.confidence:.4f}",
            "frame_index": event.frame_index,
            "video_time_seconds": f"{event.video_time_seconds:.3f}",
            "bbox_x1": f"{event.bbox[0]:.1f}",
            "bbox_y1": f"{event.bbox[1]:.1f}",
            "bbox_x2": f"{event.bbox[2]:.1f}",
            "bbox_y2": f"{event.bbox[3]:.1f}",
            "coverage_ratio": f"{event.coverage_ratio:.6f}",
            "status": VIOLATION,
            "reason": REASON_FULL_INSIDE,
        }
        self._writer.writerow(row)
        self._fh.flush()
        self.events_written += 1
        print(
            f"[EVENT] {event_id} frame={event.frame_index} t={event.video_time_seconds:.2f}s "
            f"track={event.track_id} {event.class_name} conf={event.confidence:.2f}"
        )

    def finish(self, status: str, frames_processed: int, seconds: float, decision_counts: dict[str, int]) -> None:
        avg_fps = frames_processed / seconds if seconds > 0 else 0.0
        summary = {
            "status": status,
            "frames_processed": frames_processed,
            "events": self.events_written,
            "decision_counts": decision_counts,
            "processing_seconds": round(seconds, 2),
            "avg_processing_fps": round(avg_fps, 2),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self._fh.close()
