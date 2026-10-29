from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .area_rules import (
    IGNORED_NO_ID,
    IGNORED_OCCLUDED,
    IGNORED_TRUNCATED,
    OUTSIDE,
    SKIPPED,
    VIOLATION,
    Decision,
)
from .config import RunConfig

STATUS_COLOR = {
    VIOLATION: (0, 0, 255),
    OUTSIDE: (255, 255, 255),
    IGNORED_TRUNCATED: (0, 165, 255),
    IGNORED_OCCLUDED: (0, 230, 230),
    IGNORED_NO_ID: (200, 120, 255),
    SKIPPED: (128, 128, 128),
}


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int | None


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Video '{path}' tidak bisa dibuka OpenCV (codec/rusak?)")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Video '{path}': resolusi tidak terbaca (file rusak?)")
    if fps <= 0.0 or not np.isfinite(fps):
        cap.release()
        raise ValueError(f"Video '{path}': FPS tidak valid ({fps}); video FPS variabel tidak didukung (FR01)")
    return cap, VideoMeta(width, height, fps, count if count > 0 else None)


def open_writer(path: Path, meta: VideoMeta) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        raise ValueError(f"Gagal membuka video writer {path}: codec mp4v tidak tersedia")
    return writer


def annotate(
    frame: np.ndarray,
    cfg: RunConfig,
    decisions: list[Decision],
    frame_index: int,
    video_time: float,
) -> None:
    """Gambar area, bbox beserta status+alasan (teks, bukan warna saja), dan header waktu (FR04)."""
    # Zona fill merah muda transparan: bbox utuh di dalamnya = pelanggaran (digambar sebelum bbox).
    vio_pts = np.int32(np.round(cfg.violation_area.points))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [vio_pts], (203, 192, 255))
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.polylines(frame, [np.int32(np.round(cfg.main_area.points))], True, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(frame, [vio_pts], True, (0, 0, 255), 2, cv2.LINE_AA)
    vio = cfg.violation_area.points
    cv2.putText(frame, "violation_area", (int(vio[2][0]) - 190, int(vio[2][1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    main = cfg.main_area.points
    cv2.putText(frame, "main_area", (int(main[2][0]) - 120, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for d in decisions:
        x1, y1, x2, y2 = d.bbox
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        color = STATUS_COLOR.get(d.status, (255, 60, 220))
        rounded = (max(0, int(round(x1))), max(0, int(round(y1))), max(0, int(round(x2))), max(0, int(round(y2))))
        cv2.rectangle(frame, rounded[:2], rounded[2:], color, 2)
        label = f"id{d.track_id} {d.class_name} {d.confidence:.2f} {d.status} {d.reason}"
        ty = rounded[1] - 6 if rounded[1] > 18 else rounded[3] + 18
        cv2.putText(frame, label, (rounded[0], ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"frame {frame_index} t={video_time:.2f}s", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
