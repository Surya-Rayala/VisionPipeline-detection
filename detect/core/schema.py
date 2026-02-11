from __future__ import annotations

"""
detect.core.schema
------------------

Canonical detection schema + base detector interface.

Schema version: det-v1 (stable)

Notes:
- Frame indices are **0-based** everywhere in this package.
- Frame filenames use zero-padding (default 6) and are derived from the 0-based index,
  e.g. frame 0 -> "000000.jpg".
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


SCHEMA_VERSION = "det-v1"


# ---------------------------
# Canonical detection types
# ---------------------------

class Detection(TypedDict, total=False):
    """Single detection in xyxy space with optional extras.

    Required keys:
      - bbox: [x1, y1, x2, y2] (float)
      - score: float
      - class_id: int
      - det_ind: int  # index within the frame (assigned by the runner)

    Optional keys:
      - class_name: str
      - keypoints: List[List[float]]  # per keypoint: [x, y, score]
      - segments: List[List[List[float]]]  # list of polygons; each polygon: [[x,y], ...]
    """

    bbox: List[float]
    score: float
    class_id: int
    det_ind: int
    class_name: Optional[str]
    keypoints: Optional[List[List[float]]]
    segments: Optional[List[List[List[float]]]]


class FrameRecord(TypedDict):
    """Single frame record in det-v1 JSON."""
    frame: int          # 0-based frame index
    file: str           # file name relative to frames_dir, e.g., "000000.jpg"
    detections: List[Detection]


class VideoMeta(TypedDict):
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


class DetectorConfig(TypedDict, total=False):
    """Detector configuration stored in JSON."""
    name: str
    backend: str                 # e.g. "ultralytics" (optional but recommended)
    weights: str                 # resolved weights path or identifier
    model_key: str               # registry key, if used (optional)
    classes: Optional[List[int]]
    conf_thresh: float
    imgsz: int
    device: str
    half: bool


# ---------------------------
# Abstract detector interface
# ---------------------------

class BaseDetector(ABC):
    """Abstract base class for detectors.

    Implementations must implement process_frame() returning a list of Detection dicts
    WITHOUT the 'det_ind' key (runner assigns per-frame det_ind).
    """

    backend: str = "unknown"  # override in concrete detectors (e.g. "ultralytics")

    def __init__(
        self,
        *,
        weights: Union[str, Path],
        conf: float = 0.25,
        classes: Optional[List[int]] = None,
        imgsz: int = 640,
        device: str = "auto",
        half: bool = False,
    ) -> None:
        self.weights = str(weights)
        self.conf = float(conf)
        self.classes = classes
        self.imgsz = int(imgsz)
        self.device_str = device
        self.device = select_device(device)
        self.half = bool(half)

    @property
    def name(self) -> str:
        """Human-readable detector name used in JSON (e.g. 'yolo_bbox')."""
        return self.__class__.__name__.lower()

    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run the detector on a single BGR frame and return detections.

        Must return a list of dicts with at least keys:
          - bbox: [x1, y1, x2, y2]
          - score: float
          - class_id: int
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: run a quick forward pass to initialize kernels/caches."""
        return None


# ---------------------------
# Shared helpers
# ---------------------------

_DEF_DEVICE_WARN = (
    "[warn] PyTorch not available; using CPU-like behavior. Install torch for acceleration."
)


def select_device(pref: str = "auto"):
    """Select a torch.device from a flexible string.

    Accepts: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', ..., 'mps'.
    Falls back gracefully if the requested backend is unavailable.
    Returns a torch.device when torch is available; otherwise returns the string 'cpu'.
    """
    p = (pref or "auto").lower().strip()

    if torch is None:
        if p not in {"auto", "cpu"}:
            print(_DEF_DEVICE_WARN)
        return "cpu"

    # Explicit selections first
    if p == "cpu":
        return torch.device("cpu")

    if p.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                return torch.device(p)
            except Exception:
                return torch.device("cuda")
        print("[warn] CUDA requested but not available; falling back to auto.")

    if p == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available; falling back to auto.")

    # 'auto' or anything else -> best available
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_classes(s: Optional[str]) -> Optional[List[int]]:
    """Parse a comma/semicolon separated list of integers into a list, or None."""
    if not s:
        return None
    out: List[int] = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or None


def frame_file_name(frame_idx_0based: int, pad: int = 6, ext: str = ".jpg") -> str:
    """Return a standardized frame file name like '000000.jpg' for 0-based indices."""
    return f"{frame_idx_0based:0{pad}d}{ext}"


__all__ = [
    "SCHEMA_VERSION",
    "Detection",
    "FrameRecord",
    "VideoMeta",
    "DetectorConfig",
    "BaseDetector",
    "select_device",
    "parse_classes",
    "frame_file_name",
]