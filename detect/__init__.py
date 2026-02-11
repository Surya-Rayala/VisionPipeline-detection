"""
detect
======
detect/
  __init__.py
  core/
    schema.py          # Detection types + det-v1 helpers
    run.py             # detect_video core logic (no CLI)
    artifacts.py       # artifact saving control, result object
    viz.py             # drawing helpers (optional dependency on cv2)
  backends/
    __init__.py        # plugin registry
    ultralytics/
      __init__.py
      detectors.py     # bbox/pose/seg wrappers
      export.py        # ultralytics exporter adapter
      registry.json    # ultralytics-specific model keys (optional)
  registry/
    registry.py        # merge registries + resolve_weights_path
    default.json       # your current registry (or split by backend)
  cli/
    detect_video.py    # argparse → calls core.run.detect_video()
    export_model.py    # argparse → calls backend exporter

A modular object-detection framework centered around a stable JSON output schema
(det-v1), with pluggable model backends (Ultralytics YOLO today; others later).

Public API (stable-ish):
- Detection schema types and helpers (core.schema)
- Detection runner (core.run.detect_video)
- Artifact control + result object (core.artifacts)
- Backend registry + detector factory (backends)
"""

from .core.schema import (
    SCHEMA_VERSION,
    Detection,
    FrameRecord,
    VideoMeta,
    DetectorConfig,
    BaseDetector,
    select_device,
    parse_classes,
    frame_file_name,
)

from .core.artifacts import (
    ArtifactOptions,
    DetectResult,
)

from .core.run import (
    detect_video,
)

from .backends import (
    create_detector,
    available_detectors,
    available_models,
)

__all__ = [
    # Schema / types
    "SCHEMA_VERSION",
    "Detection",
    "FrameRecord",
    "VideoMeta",
    "DetectorConfig",
    "BaseDetector",
    "select_device",
    "parse_classes",
    "frame_file_name",
    # Running + outputs
    "ArtifactOptions",
    "DetectResult",
    "detect_video",
    # Backends
    "create_detector",
    "available_detectors",
    "available_models",
]