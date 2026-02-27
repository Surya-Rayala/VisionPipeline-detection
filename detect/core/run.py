from __future__ import annotations

"""
detect.core.run
---------------

Core detection runner (no CLI).

Key behavior:
- Always returns the det-v1 JSON payload in memory (DetectResult.payload).
- Artifact saving is fully controlled via ArtifactOptions.
- If no artifacts are requested (save_json/save_frames/save_video all False),
  it will NOT create out_dir/run_dir.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .schema import (
    SCHEMA_VERSION,
    Detection,
    FrameRecord,
    VideoMeta,
    DetectorConfig,
    PromptSpec,
    frame_file_name,
)
from .artifacts import ArtifactOptions, DetectResult, compute_run_dirs
from .viz import draw_detections
from ..backends import create_detector


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for video I/O (cv2 import failed).")


def _video_meta(cap, src_path: Path) -> Tuple[VideoMeta, float, Tuple[int, int]]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta: VideoMeta = {
        "path": str(src_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    return meta, fps, (width, height)


def _open_writer(path: Path, fps: float, size: Tuple[int, int], fourcc: str = "mp4v"):
    code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(path), code, fps, size)
    if not writer.isOpened():
        print(f"[warn] Failed to open video writer at {path}, skipping save_video.")
        return None
    return writer


def detect_video(
    *,
    video: Union[str, Path],
    detector: str,
    weights: Union[str, Path],
    classes: Optional[List[int]] = None,
    conf_thresh: float = 0.25,
    imgsz: int = 640,
    device: str = "auto",
    half: bool = False,
    task: str = "auto",
    prompts: Optional[PromptSpec] = None,
    topk: Optional[int] = None,
    models_dir: Union[str, Path] = "models",
    download_models: bool = True,
    artifacts: Optional[ArtifactOptions] = None,
    # --- unified convenience toggles (all opt-in; default is "save nothing") ---
    save_json: Optional[bool] = None,
    save_frames: Optional[bool] = None,
    save_video: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
    run_name: Optional[str] = None,
    progress: Optional[bool] = None,
    display: Optional[bool] = None,
) -> DetectResult:
    """
    Run detection on a video and return a DetectResult.

    - In-memory payload is always present in DetectResult.payload.
    - Any saved artifact paths appear in DetectResult.paths.

    Artifacts are controlled via `artifacts`:
      - save_json: write detections.json
      - save_frames: write frames/*.jpg
      - save_video: write annotated video
      - display: show live window (press 'q' to quit early)
      - task/prompts/topk control detector behavior for promptable/open-vocab/classify tasks.

    If no artifacts are requested, no directories are created.

    You may also pass convenience kwargs (recommended for simple usage):
      - save_json: bool
      - save_frames: bool
      - save_video: filename (e.g., "annotated.mp4")
      - out_dir: output root (only used if saving)
      - run_name: run folder name (only used if saving)
      - progress: enable/disable tqdm (defaults to ArtifactOptions)

    Do not pass both `artifacts=` and any convenience kwargs.
    """
    _require_cv2()

    # --- ArtifactOptions is the canonical internal representation ---
    # By default, nothing is saved unless explicitly requested.
    convenience_used = any(
        x is not None for x in (save_json, save_frames, save_video, out_dir, run_name, progress, display)
    )
    if artifacts is not None and convenience_used:
        raise TypeError(
            "Pass either `artifacts=...` OR convenience args (save_json/save_frames/save_video/out_dir/run_name/progress), not both."
        )

    if artifacts is None:
        artifacts = ArtifactOptions()

        if save_json is not None:
            artifacts.save_json = bool(save_json)
        if save_frames is not None:
            artifacts.save_frames = bool(save_frames)

        if save_video is not None:
            name = str(save_video).strip()
            if name:
                artifacts.save_video = True
                artifacts.save_video_name = name
            else:
                artifacts.save_video = False

        if out_dir is not None:
            artifacts.out_dir = str(out_dir)
        if run_name is not None:
            artifacts.run_name = str(run_name)
        if progress is not None:
            artifacts.progress = bool(progress)
        if display is not None:
            artifacts.display = bool(display)

    src_path = Path(video)
    if not src_path.exists():
        raise FileNotFoundError(f"Video not found: {src_path}")

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_path}")

    meta, fps, (w, h) = _video_meta(cap, src_path)

    # Create detector
    det = create_detector(
        name=detector,
        weights=weights,
        conf=conf_thresh,
        classes=classes,
        imgsz=imgsz,
        device=device,
        half=half,
        task=task,
        prompts=prompts,
        topk=topk,
        models_dir=models_dir,
        allow_download=download_models,
    )

    # Compute run dirs but do not create unless saving something
    run_dir: Optional[Path] = None
    frames_dir: Optional[Path] = None
    json_path: Optional[Path] = None
    video_path: Optional[Path] = None

    wants_any = artifacts.wants_any_artifact()
    if wants_any:
        dirs = compute_run_dirs(video_path=src_path, detector_name=detector, artifacts=artifacts)
        run_dir = dirs["run_dir"]
        frames_dir = dirs["frames_dir"]

        # Create run_dir immediately (we might save JSON or video even if no frames)
        run_dir.mkdir(parents=True, exist_ok=True)
        if artifacts.save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        if artifacts.save_json:
            json_path = run_dir / "detections.json"
        if artifacts.save_video:
            video_path = run_dir / artifacts.save_video_name

    # Progress bar (optional)
    pbar = None
    if artifacts.progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=int(meta.get("frame_count", 0)) or None, desc="Detect", disable=False)
        except Exception:
            pbar = None

    # Video writer (created lazily once we have first frame)
    writer = None

    records: List[FrameRecord] = []
    frames_processed = 0
    quit_early = False

    try:
        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frames_processed += 1

            # Run detector
            dets = det.process_frame(frame)

            # Assign det_ind sequentially for this frame
            for i, d in enumerate(dets):
                d["det_ind"] = i

            # Frame file name (0-based)
            rel_name = frame_file_name(
                frame_idx,
                pad=artifacts.frame_pad,
                ext=artifacts.frame_ext,
            )

            # Save raw frame image if requested
            if artifacts.save_frames and frames_dir is not None:
                # set JPEG quality if applicable
                params = []
                if artifacts.frame_ext.lower() in (".jpg", ".jpeg"):
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(artifacts.jpg_quality)]
                cv2.imwrite(str(frames_dir / rel_name), frame, params)

            # Build frame record (always in payload)
            rec: FrameRecord = {
                "frame": frame_idx,
                "file": rel_name,
                "detections": dets,  # type: ignore
            }
            records.append(rec)

            # Annotate/display if requested
            if artifacts.save_video or artifacts.display:
                vis = draw_detections(frame, dets, frame_idx)

                if artifacts.save_video and video_path is not None:
                    if writer is None:
                        writer = _open_writer(video_path, fps=fps, size=(w, h), fourcc=artifacts.fourcc)
                    if writer is not None:
                        writer.write(vis)

                if artifacts.display:
                    cv2.imshow("Detect", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[info] Quit requested by user.")
                        quit_early = True
                        break

            if pbar is not None:
                pbar.update(1)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if artifacts.display:
            cv2.destroyAllWindows()
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    # Detector config stored in JSON (minimal, meaningful)
    det_cfg: DetectorConfig = {
        "name": detector,
        "backend": getattr(det, "backend", "unknown"),
        "weights": str(getattr(det, "weights", weights)),
        "classes": classes,
        "conf_thresh": float(conf_thresh),
        "imgsz": int(imgsz),
        "device": str(device),
        "half": bool(half),
        "task": str(task),
    }
    if prompts is not None:
        det_cfg["prompts"] = prompts  # type: ignore
    if topk is not None:
        det_cfg["topk"] = int(topk)  # type: ignore

    # frames_dir in payload:
    # - if frames are saved, it's the actual path
    # - if frames are not saved, keep a meaningful placeholder (run_dir/frames if run_dir exists, else "")
    frames_dir_str = ""
    if wants_any and run_dir is not None:
        frames_dir_str = str((frames_dir or (run_dir / artifacts.frames_subdir)).resolve())

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "video": meta,
        "detector": det_cfg,
        "frames_dir": frames_dir_str,
        "frames": records,
    }

    # Save JSON only if requested
    paths: Dict[str, str] = {}
    if artifacts.save_json and json_path is not None:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json_path"] = str(json_path.resolve())

    if artifacts.save_frames and frames_dir is not None:
        paths["frames_dir"] = str(frames_dir.resolve())

    if artifacts.save_video and video_path is not None:
        paths["video_path"] = str(video_path.resolve())

    if wants_any and run_dir is not None:
        paths["run_dir"] = str(run_dir.resolve())

    return DetectResult(payload=payload, paths=paths, frames_processed=frames_processed, quit_early=quit_early)


__all__ = ["detect_video"]