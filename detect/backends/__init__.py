from __future__ import annotations

"""
detect.backends
---------------

Backend/plugin registry.

Today we ship an Ultralytics backend with three detectors:
- yolo_bbox
- yolo_pose
- yolo_seg

This module provides:
- register_detector() for future backends
- create_detector() as the main factory
- available_detectors() listing
- available_models() combining registry + installed models

Design notes:
- Detector keys are currently simple (e.g. "yolo_bbox") for backward compatibility.
- Internally, we store (backend, name) so we can support future keys like "onnxrt:foo".
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from ..core.schema import BaseDetector
from ..registry.registry import (
    list_installed_models,
    list_registered_models,
    resolve_weights_path,
)

# Import and register built-in backend(s)
from .ultralytics.detectors import (  # noqa: E402
    YOLOBBoxDetector,
    YOLOPoseDetector,
    YOLOSegDetector,
)


@dataclass(frozen=True)
class DetectorSpec:
    backend: str
    name: str
    cls: Type[BaseDetector]


# Keyed by detector "public name" (e.g. "yolo_bbox") -> spec
_DETECTORS: Dict[str, DetectorSpec] = {}


def register_detector(*, name: str, backend: str, cls: Type[BaseDetector]) -> None:
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Detector name must be non-empty.")
    if key in _DETECTORS:
        raise ValueError(f"Detector '{key}' already registered.")
    _DETECTORS[key] = DetectorSpec(backend=backend, name=key, cls=cls)


def available_detectors() -> List[str]:
    return sorted(_DETECTORS.keys())


def available_models(
    *,
    detector: Optional[str] = None,
    backend: Optional[str] = None,
    models_dir: Union[str, Path] = "models",
) -> dict:
    """
    Return a dict: {"registered": {...}, "installed": {...}}.

    - registered models are from the JSON registry files (optionally filtered).
    - installed models are local model files found under models_dir.
    """
    det_key = detector.strip().lower() if detector else None
    return {
        "registered": list_registered_models(detector=det_key, backend=backend),
        "installed": list_installed_models(models_dir=models_dir),
    }


def create_detector(
    *,
    name: str,
    weights: Union[str, Path],
    conf: float = 0.25,
    classes: Optional[List[int]] = None,
    imgsz: int = 640,
    device: str = "auto",
    half: bool = False,
    models_dir: Union[str, Path] = "models",
    allow_download: bool = True,
) -> BaseDetector:
    """
    Create a detector instance.

    - Resolves weights from local path / URL / registry key into a local file.
    - Instantiates the registered detector class.
    """
    key = (name or "").strip().lower()
    if key not in _DETECTORS:
        raise ValueError(f"Unknown detector '{name}'. Available: {', '.join(available_detectors())}")

    spec = _DETECTORS[key]

    weights_path = resolve_weights_path(
        weights,
        models_dir=models_dir,
        detector=key,
        backend=spec.backend,
        allow_download=allow_download,
    )
    return spec.cls(
        weights=weights_path,
        conf=conf,
        classes=classes,
        imgsz=imgsz,
        device=device,
        half=half,
    )


# -------------------------
# Built-in detector wiring
# -------------------------
register_detector(name="yolo_bbox", backend="ultralytics", cls=YOLOBBoxDetector)
register_detector(name="yolo_pose", backend="ultralytics", cls=YOLOPoseDetector)
register_detector(name="yolo_seg", backend="ultralytics", cls=YOLOSegDetector)


__all__ = [
    "register_detector",
    "available_detectors",
    "available_models",
    "create_detector",
]