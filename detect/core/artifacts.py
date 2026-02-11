from __future__ import annotations

"""
detect.core.artifacts
---------------------

Artifact saving controls and the result object returned by detect_video().

Key design goals:
- All artifacts are optional (JSON, frames, annotated video).
- If nothing is being saved, no output directories should be created.
- When imported/used as a library, the in-memory JSON payload is always returned.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ArtifactOptions:
    """
    Controls what artifacts are written to disk.

    If all of (save_json, save_frames, save_video) are False, the runner will not
    create any out_dir/run_dir and will operate in "in-memory only" mode.

    Paths:
      - out_dir/run_name define where artifacts go when any save_* is enabled.
      - save_video is a filename (e.g. "annotated.mp4") saved inside run_dir.
    """

    # Primary toggles
    save_json: bool = False
    save_frames: bool = False
    save_video: bool = False

    # Where to save (only used if any save_* is True)
    out_dir: Path = Path("out")
    run_name: Optional[str] = None

    # Frames
    frames_subdir: str = "frames"
    frame_pad: int = 6
    frame_ext: str = ".jpg"
    jpg_quality: int = 95  # OpenCV JPEG quality

    # Annotated video
    save_video_name: str = "detect_annotated.mp4"
    fourcc: str = "mp4v"

    # Progress / UX
    display: bool = False
    progress: bool = True

    def wants_any_artifact(self) -> bool:
        return bool(self.save_json or self.save_frames or self.save_video)


@dataclass
class DetectResult:
    """
    Returned by detect_video(). Always contains the in-memory det-v1 payload,
    plus any saved artifact paths that were requested.

    `paths` keys are only present when that artifact is saved.
    """
    payload: Dict[str, Any]
    paths: Dict[str, str] = field(default_factory=dict)

    # Useful runtime stats (optional)
    frames_processed: int = 0
    quit_early: bool = False


def compute_run_dirs(
    *,
    video_path: Path,
    detector_name: str,
    artifacts: ArtifactOptions,
) -> Dict[str, Path]:
    """
    Compute run_dir and frames_dir, but do NOT create them.

    Returns:
      {"run_dir": Path, "frames_dir": Path}
    """
    run_name = artifacts.run_name or f"{video_path.stem}_{detector_name}"
    run_dir = artifacts.out_dir / run_name
    frames_dir = run_dir / artifacts.frames_subdir
    return {"run_dir": run_dir, "frames_dir": frames_dir}


__all__ = [
    "ArtifactOptions",
    "DetectResult",
    "compute_run_dirs",
]