"""Model dimuat satu kali; per frame deteksi YOLO + track id ByteTrack (persist)."""

from pathlib import Path

from .area_rules import Detection

TRACKER_DEFAULT = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


class Detector:
    def __init__(self, weights: str | Path, *, conf: float, imgsz: int, device: str, track_buffer: int):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self._call_args = dict(conf=conf, imgsz=imgsz, device=device, persist=True, verbose=False)
        self._track_buffer = track_buffer
        self._tracker: str | None = None

    def track_frame(self, frame, workdir: Path) -> list[Detection]:
        if self._tracker is None:
            yaml_path = Path(workdir) / "tracker_bytetrack.yaml"
            if not yaml_path.exists():
                import yaml

                yaml_path.write_text(yaml.safe_dump(dict(TRACKER_DEFAULT, track_buffer=self._track_buffer)))
            self._tracker = str(yaml_path)
        results = self.model.track(frame, tracker=self._tracker, **self._call_args)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        detections: list[Detection] = []
        for bbox, cls_id, conf, tid in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), ids):
            detections.append(
                Detection(
                    class_name=self.names.get(int(cls_id)),
                    confidence=float(conf),
                    track_id=int(tid) if tid is not None else None,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
            )
        return detections
