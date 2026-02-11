from __future__ import annotations

"""
detect.cli.detect_video
-----------------------

CLI wrapper for detect.core.run.detect_video.

Defaults:
- Does NOT save JSON or frames unless you pass --json and/or --frames.
- Does NOT create out_dir/run_dir unless any artifact is being saved.
- Always prints the in-memory det-v1 payload to stdout.

Examples:
  # In-memory only (no files written):
  python -m detect.cli.detect_video --video in.mp4 --detector yolo_bbox --weights yolo26n

  # Save JSON + frames:
  python -m detect.cli.detect_video --video in.mp4 --detector yolo_bbox --weights yolo26n --json --frames --out-dir out --run-name run1

  # Save annotated video (and optionally json/frames):
  python -m detect.cli.detect_video --video in.mp4 --detector yolo_seg --weights yolo26n-seg --save-video annotated.mp4 --out-dir out --run-name seg1
"""

import argparse
import json
from pathlib import Path

from ..backends import available_detectors, available_models
from ..core.artifacts import ArtifactOptions
from ..core.run import detect_video
from ..core.schema import parse_classes


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a detector on a video and output det-v1 JSON + optional artifacts.")
    ap.add_argument("--video", type=Path, required=False, help="Path to input video")
    ap.add_argument("--detector", choices=available_detectors(), required=False, help="Detector backend (task wrapper)")
    ap.add_argument("--weights", type=str, required=False, help="Weights path/URL/registry key (e.g., models/yolo26n.pt or yolo26n)")
    ap.add_argument("--classes", type=str, default=None, help='Comma/semicolon-separated class ids, e.g. "0,2"')
    ap.add_argument("--conf-thresh", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--half", action="store_true")

    # Artifact controls (opt-in)
    ap.add_argument("--json", action="store_true", help="Write detections.json into run dir")
    ap.add_argument("--frames", action="store_true", help="Write frames/*.jpg into run dir")
    ap.add_argument("--save-video", type=str, default=None, help="Save annotated video filename inside run dir (e.g., detect_annotated.mp4)")
    ap.add_argument("--display", action="store_true", help="Show live window (press q to quit)")

    ap.add_argument("--out-dir", type=Path, default=Path("out"), help="Root output directory (only used if saving artifacts)")
    ap.add_argument("--run-name", type=str, default=None, help="Run folder name inside out-dir")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    ap.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory where model files are stored/downloaded")
    ap.add_argument("--no-download", action="store_true", help="Disable automatic download of weights from registry URLs")
    ap.add_argument("--list-models", action="store_true", help="List registered and installed models, then exit")
    return ap


def run_cli() -> None:
    ap = _build_argparser()
    args = ap.parse_args()

    if args.list_models:
        info = available_models(detector=None, models_dir=args.models_dir)
        print(json.dumps(info, indent=2))
        return

    missing = []
    for flag, val in (("--video", args.video), ("--detector", args.detector), ("--weights", args.weights)):
        if val is None:
            missing.append(flag)
    if missing:
        ap.error("Missing required arguments: " + " ".join(missing))

    classes_list = parse_classes(args.classes) if args.classes is not None else None

    artifacts = ArtifactOptions(
        save_json=bool(args.json),
        save_frames=bool(args.frames),
        save_video=bool(args.save_video),
        out_dir=args.out_dir,
        run_name=args.run_name,
        save_video_name=args.save_video or "detect_annotated.mp4",
        display=bool(args.display),
        progress=not args.no_progress,
    )

    res = detect_video(
        video=args.video,
        detector=args.detector,
        weights=args.weights,
        classes=classes_list,
        conf_thresh=args.conf_thresh,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        models_dir=args.models_dir,
        download_models=not args.no_download,
        artifacts=artifacts,
    )

    # Always print the in-memory payload for CLI usage (useful even when saving)
    print(json.dumps(res.payload, indent=2))


if __name__ == "__main__":
    run_cli()


__all__ = ["run_cli"]