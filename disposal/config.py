import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Area:
    """Polygon area pada resolusi referensi (float32, n>=3 titik)."""

    name: str
    points: np.ndarray


@dataclass(frozen=True)
class RunConfig:
    reference_width: int
    reference_height: int
    main_area: Area
    violation_area: Area
    confidence_threshold: float
    imgsz: int
    target_classes: list[str]
    occluder_classes: list[str]
    frame_edge_margin_px: int
    max_missing_frames: int
    device: str


def _validated_area(raw: object, name: str, width: int, height: int) -> Area:
    if not isinstance(raw, dict) or "points" not in raw:
        raise ValueError(f"Area '{name}': butuh objek dengan kolom 'points'")
    try:
        pts = np.asarray(raw["points"], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Area '{name}': 'points' harus daftar [[x,y], ...]") from exc
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError(f"Area '{name}': minimal 3 titik [x,y] (didapat shape {tuple(pts.shape)})")
    if (pts < 0).any() or (pts[:, 0] > width).any() or (pts[:, 1] > height).any():
        raise ValueError(f"Area '{name}': titik berada di luar frame referensi {width}x{height}")
    if cv2.contourArea(pts) <= 0:
        raise ValueError(f"Area '{name}': luas harus > 0 (titik menulis ulang/garis lurus)")
    # Containment memakai intersectConvexConvex (bbox selalu convex) -> area wajib convex.
    p = np.vstack([pts, pts[:1]])
    e = p[1:] - p[:-1]
    cr = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
    if not ((cr >= -1e-6).all() or (cr <= 1e-6).all()):
        raise ValueError(f"Area '{name}': polygon cekung/berpotongan tidak didukung; gunakan polygon convex")
    return Area(name=name, points=np.ascontiguousarray(pts))


def load_config(path: Path) -> RunConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Tidak bisa membaca config '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Config harus berupa objek JSON")
    try:
        width, height = int(raw["reference_width"]), int(raw["reference_height"])
        conf = float(raw.get("confidence_threshold", 0.25))
        imgsz = int(raw.get("imgsz", 640))
        targets = list(raw.get("target_classes", []))
        occluders = list(raw.get("occluder_classes", []))
        margin = int(raw.get("frame_edge_margin_px", 2))
        missing = int(raw.get("max_missing_frames", 30))
        device = str(raw.get("device", "auto"))
        areas = raw.get("areas", {})
        main = _validated_area(areas.get("main_area"), "main_area", width, height)
        vio = _validated_area(areas.get("violation_area"), "violation_area", width, height)
    except KeyError as exc:
        raise ValueError(f"Config kurang kolom wajib: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config tidak valid: {exc}") from exc
    if width <= 0 or height <= 0 or imgsz <= 0:
        raise ValueError("reference_width/reference_height/imgsz harus > 0")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"confidence_threshold harus 0..1 (didapat {conf})")
    if margin < 0 or missing < 1:
        raise ValueError("frame_edge_margin_px >= 0 dan max_missing_frames >= 1 wajib")
    if not targets:
        raise ValueError("target_classes tidak boleh kosong")
    return RunConfig(
        reference_width=width,
        reference_height=height,
        main_area=main,
        violation_area=vio,
        confidence_threshold=conf,
        imgsz=imgsz,
        target_classes=targets,
        occluder_classes=occluders,
        frame_edge_margin_px=margin,
        max_missing_frames=missing,
        device=device,
    )
