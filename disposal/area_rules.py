import cv2
import numpy as np
from dataclasses import dataclass

VIOLATION = "VIOLATION"
OUTSIDE = "OUTSIDE"
IGNORED_TRUNCATED = "IGNORED_TRUNCATED"
IGNORED_OCCLUDED = "IGNORED_OCCLUDED"
IGNORED_NO_ID = "IGNORED_NO_ID"
SKIPPED = "SKIPPED"

REASON_FULL_INSIDE = "FULL_BBOX_INSIDE"
REASON_OUTSIDE = "BBOX_NOT_FULLY_INSIDE"
REASON_FRAME_EDGE = "FRAME_EDGE_TOUCH"
REASON_OVERLAP = "BBOX_OVERLAP"
REASON_NO_TRACK_ID = "PENDING_TRACK_ID"
REASON_NOT_TARGET = "CLASS_NOT_TARGET"
REASON_INVALID_BBOX = "INVALID_BBOX"

# Toleransi numerik containment (PRD §3): kebocoran <=1% luas bbox masih dianggap di dalam,
# supaya bbox yang mepet garis polygon tidak flap keluar-masuk.
_CONTAIN_EPS = 1e-2


@dataclass(frozen=True)
class Detection:
    """Satu deteksi frame-native (kontrak data minimum PRD §5)."""

    class_name: str | None
    confidence: float
    track_id: int | None
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Decision:
    """Keputusan satu deteksi pada satu frame (status, reason, coverage_ratio, pengecualian)."""

    track_id: int | None
    class_name: str | None
    confidence: float
    bbox: tuple[float, float, float, float]
    status: str
    reason: str
    coverage_ratio: float
    frame_index: int


@dataclass(frozen=True)
class Event:
    """Satu episode masuk area (PRD §4) -> satu baris CSV."""

    track_id: int
    class_name: str | None
    confidence: float
    frame_index: int
    video_time_seconds: float
    bbox: tuple[float, float, float, float]
    coverage_ratio: float


@dataclass
class TrackState:
    """Penanda episode per track id."""

    last_seen_frame: int
    inside: bool = False
    event_written: bool = False


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _quad(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    """Luas irisan dua polygon convex; poligon saling bersentuhan garis/titik -> 0."""
    area, _ = cv2.intersectConvexConvex(a, b)
    return max(0.0, float(area))


def _decision(d: Detection, status: str, reason: str, coverage: float, frame_index: int) -> Decision:
    return Decision(d.track_id, d.class_name, d.confidence, d.bbox, status, reason, coverage, frame_index)


class AreaRules:
    """Area rules (PRD §5): pengecualian, keputusan, dan state episode."""

    def __init__(
        self,
        violation_area,
        *,
        target_classes,
        occluder_classes,
        margin_px: int,
        max_missing_frames: int,
    ):
        self.area = violation_area
        self.targets = set(target_classes)
        self.occluders = set(occluder_classes)
        self.margin = margin_px
        self.max_missing = max_missing_frames
        self.states: dict[int, TrackState] = {}

    def _truncated(self, bbox, frame_size: tuple[int, int]) -> bool:
        """EX01: bbox menyentuh/melewati batas frame dalam margin (diperiksa sebelum clipping)."""
        w, h = frame_size
        x1, y1, x2, y2 = bbox
        return x1 <= self.margin or y1 <= self.margin or x2 >= w - self.margin or y2 >= h - self.margin

    def _coverage(self, bbox) -> float:
        """coverage_ratio = luas(B ∩ P) / luas(B) — bukan IoU (PRD §3)."""
        denom = bbox_area(bbox)
        if denom <= 0.0:
            return 0.0
        return _intersection_area(self.area.points, _quad(bbox)) / denom

    @staticmethod
    def _occluded(bbox, occluder_boxes) -> bool:
        """EX02: irisan luas > 0 dengan bbox kendaraan lain; garis/titik tidak memenuhi."""
        quad = _quad(bbox)
        return any(_intersection_area(quad, _quad(other)) > 1e-6 for other in occluder_boxes)

    def evaluate(
        self,
        detections: list[Detection],
        frame_index: int,
        frame_size: tuple[int, int],
        video_time: float,
    ) -> tuple[list[Decision], list[Event]]:
        # Track kedaluwarsa dibuang: kemunculan lagi = episode baru (dedup hanya selama id hidup).
        for tid in list(self.states):
            if frame_index - self.states[tid].last_seen_frame > self.max_missing:
                del self.states[tid]
        occluder_boxes = [d.bbox for d in detections if d.class_name in self.occluders and bbox_area(d.bbox) > 0]
        decisions: list[Decision] = []
        events: list[Event] = []
        for d in detections:
            if bbox_area(d.bbox) <= 0:
                decisions.append(_decision(d, SKIPPED, REASON_INVALID_BBOX, 0.0, frame_index))
                continue
            state = self.states.get(d.track_id) if d.track_id is not None else None
            is_target = d.class_name in self.targets
            cov = self._coverage(d.bbox)
            if self._truncated(d.bbox, frame_size):
                status, reason = IGNORED_TRUNCATED, REASON_FRAME_EDGE
            elif self._occluded(d.bbox, [b for b in occluder_boxes if b != d.bbox]):
                status, reason = IGNORED_OCCLUDED, REASON_OVERLAP
            elif is_target and cov < 1.0 - _CONTAIN_EPS:
                status, reason = OUTSIDE, REASON_OUTSIDE
            elif is_target:
                status, reason = VIOLATION, REASON_FULL_INSIDE
            else:
                decisions.append(_decision(d, SKIPPED, REASON_NOT_TARGET, cov, frame_index))
                continue
            if d.track_id is None:
                # Tanpa track id tidak ada episode -> tidak pernah menulis event (PRD §4).
                decisions.append(_decision(d, IGNORED_NO_ID, REASON_NO_TRACK_ID, cov, frame_index))
                continue
            if state is None:
                state = self.states[d.track_id] = TrackState(last_seen_frame=frame_index)
            state.last_seen_frame = frame_index
            if cov >= 1.0 - _CONTAIN_EPS:
                # Bukti di dalam area (walau IGNORED) menandai episode; hanya OUTSIDE yang menutup.
                state.inside = True
                if status == VIOLATION and not state.event_written:
                    event = Event(int(d.track_id), d.class_name, d.confidence, frame_index, video_time, d.bbox, cov)
                    state.event_written = True
                    events.append(event)
            else:
                state.inside = False
                state.event_written = False
            decisions.append(_decision(d, status, reason, cov, frame_index))
        return decisions, events
