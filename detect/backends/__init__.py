from __future__ import annotations

"""
detect.backends
---------------

Backend/plugin registry.

Today we ship an Ultralytics backend exposed primarily as a single detector:
- ultralytics  (use `task=` to select detect/segment/pose/obb/classify/openvocab/sam...)

Backward-compatible aliases are also registered:
- yolo_bbox, yolo_pose, yolo_seg
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..core.schema import BaseDetector, PromptSpec
from ..registry.registry import (
    list_installed_models,
    list_registered_models,
    resolve_weights_path,
)

from .ultralytics.detectors import UltralyticsDetector  # noqa: E402


@dataclass(frozen=True)
class DetectorSpec:
    backend: str
    name: str
    cls: Type[BaseDetector]
    init_kwargs: Dict[str, Any]


# Keyed by detector "public name" (e.g. "yolo_bbox") -> spec
_DETECTORS: Dict[str, DetectorSpec] = {}


def register_detector(*, name: str, backend: str, cls: Type[BaseDetector], init_kwargs: Optional[Dict[str, Any]] = None) -> None:
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Detector name must be non-empty.")
    if key in _DETECTORS:
        raise ValueError(f"Detector '{key}' already registered.")
    _DETECTORS[key] = DetectorSpec(backend=backend, name=key, cls=cls, init_kwargs=dict(init_kwargs or {}))


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
    task: str = "auto",
    prompts: Optional[PromptSpec] = None,
    topk: Optional[int] = None,
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
        detector=None if key == "ultralytics" else key,
        backend=spec.backend,
        allow_download=allow_download,
    )

    init_kwargs: Dict[str, Any] = dict(spec.init_kwargs)

    # Only pass task/prompts/topk to detectors that support them; UltralyticsDetector does.
    # IMPORTANT: preserve per-detector defaults (e.g. yolo_bbox -> detect) unless the caller
    # explicitly overrides via a non-'auto' task.
    if task is not None:
        t = str(task).strip().lower()
        if t and t != "auto":
            init_kwargs["task"] = t

    if prompts is not None:
        init_kwargs["prompts"] = prompts

    if topk is not None:
        init_kwargs["topk"] = topk

    return spec.cls(
        weights=weights_path,
        conf=conf,
        classes=classes,
        imgsz=imgsz,
        device=device,
        half=half,
        **init_kwargs,
    )


# -------------------------
# Built-in detector wiring
# -------------------------
# Primary organized entrypoint
register_detector(name="ultralytics", backend="ultralytics", cls=UltralyticsDetector, init_kwargs={"task": "auto"})

# Backward-compatible aliases
register_detector(name="yolo_bbox", backend="ultralytics", cls=UltralyticsDetector, init_kwargs={"task": "detect"})
register_detector(name="yolo_pose", backend="ultralytics", cls=UltralyticsDetector, init_kwargs={"task": "pose"})
register_detector(name="yolo_seg", backend="ultralytics", cls=UltralyticsDetector, init_kwargs={"task": "segment"})


__all__ = [
    "register_detector",
    "available_detectors",
    "available_models",
    "create_detector",
]