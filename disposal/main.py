import argparse
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from disposal import video_io
from disposal.area_rules import AreaRules
from disposal.config import load_config
from disposal.detector import Detector
from disposal.reporter import Reporter


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deteksi pelanggaran area disposal pada video lokal (PRD Disposal).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true", help="timpa hasil run sebelumnya secara eksplisit")
    args = ap.parse_args(argv)

    cap = writer = reporter = None
    status, frames = "completed", 0
    started = time.monotonic()
    counts: Counter[str] = Counter()
    try:
        run_cfg = load_config(Path(args.config))
        for flag, name in (("--video", args.video), ("--model", args.model), ("--config", args.config)):
            if not Path(name).is_file():
                raise ValueError(f"{flag} '{name}' bukan file yang ada")
        cap, meta = video_io.open_video(args.video)
        if (meta.width, meta.height) != (run_cfg.reference_width, run_cfg.reference_height):
            raise ValueError(
                f"Resolusi video {meta.width}x{meta.height} != referensi config "
                f"{run_cfg.reference_width}x{run_cfg.reference_height} (FR03); "
                "sesuaikan konfigurasi atau resolusi video"
            )
        out_dir = args.output
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            raise ValueError(f"Folder output '{out_dir}' tidak kosong; pakai --overwrite bila memang menimpa")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, out_dir / "config_used.json")
        device = _resolve_device(run_cfg.device)
        print(f"[INFO] device={device} video={meta.width}x{meta.height}@{meta.fps:.4f}fps output={out_dir}")
        detector = Detector(
            args.model,
            conf=run_cfg.confidence_threshold,
            imgsz=run_cfg.imgsz,
            device=device,
            track_buffer=run_cfg.max_missing_frames,
        )
        unknown = (set(run_cfg.target_classes) | set(run_cfg.occluder_classes)) - set(detector.names.values())
        if unknown:
            raise ValueError(
                f"Kelas config tidak ada di model (FR02): {sorted(unknown)}; "
                f"nama kelas model: {sorted(detector.names.values())}"
            )
        rules = AreaRules(
            run_cfg.violation_area,
            target_classes=run_cfg.target_classes,
            occluder_classes=run_cfg.occluder_classes,
            margin_px=run_cfg.frame_edge_margin_px,
            max_missing_frames=run_cfg.max_missing_frames,
        )
        writer = video_io.open_writer(out_dir / "annotated.mp4", meta)
        reporter = Reporter(out_dir, run_id=uuid.uuid4().hex[:12], video_name=Path(args.video).name)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_time = frames / meta.fps
            dets = detector.track_frame(frame, workdir=out_dir)
            decisions, events = rules.evaluate(dets, frames, (meta.width, meta.height), video_time)
            for d in decisions:
                counts[d.status] += 1
            for e in events:
                reporter.write_event(e, f"{reporter.run_id}-{reporter.events_written + 1:04d}")
            video_io.annotate(frame, run_cfg, decisions, frames, video_time)
            writer.write(frame)
            frames += 1
            if frames % 500 == 0:
                print(
                    f"[INFO] frame {frames}/{meta.frame_count or '?'} t={video_time:.1f}s "
                    f"events={reporter.events_written} waktu={time.monotonic() - started:.0f}s"
                )
        print(f"[OK] selesai: {frames} frame, {reporter.events_written} event -> {out_dir}")
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        print(f"[INFO] Dihentikan (Ctrl+C). Hasil parsial tetap tersimpan di {args.output}", file=sys.stderr)
        return 130
    except ValueError as exc:
        status = "terminated"
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if reporter is not None:
            reporter.finish(status, frames, time.monotonic() - started, dict(counts))


if __name__ == "__main__":
    raise SystemExit(main())
