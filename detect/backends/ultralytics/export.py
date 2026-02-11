from __future__ import annotations

"""
detect.backends.ultralytics.export
---------------------------------

Ultralytics exporter adapter.

Today we only export Ultralytics YOLO `.pt` models via `YOLO(...).export(...)`.
Later we can add other exporters/backends and route based on registry backend/model type.

References:
- Ultralytics export mode supports multiple formats via `model.export(format=...)`.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ...registry.registry import resolve_weights_path, list_registered_models, list_installed_models

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore
    _ULTRA_IMPORT_ERR = e


SUPPORTED_FORMATS = {
    "torchscript", "onnx", "engine", "openvino", "coreml",
    "saved_model", "tflite", "edgetpu", "tfjs", "paddle",
    "mnn", "ncnn", "imx", "rknn",
}


def _norm_formats(s: Union[str, List[str]]) -> List[str]:
    if isinstance(s, str):
        toks = [t.strip() for t in s.replace(";", ",").split(",")]
    else:
        toks = list(s)
    return [t for t in toks if t]


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _move_artifact(artifact: Union[str, Path], dst_dir: Path) -> Union[str, None]:
    """Move file/dir into dst_dir; return new absolute path (str) or None if missing."""
    p = Path(artifact)
    if not p.exists():
        return None
    _ensure_out_dir(dst_dir)
    target = dst_dir / p.name
    # Avoid clobber: if exists, add numeric suffix
    if target.exists():
        stem, suf = target.stem, target.suffix
        i = 1
        while True:
            candidate = target.with_name(f"{stem}_{i}{suf}")
            if not candidate.exists():
                target = candidate
                break
            i += 1
    shutil.move(str(p), str(target))
    return str(target.resolve())


def export_model_ultralytics(
    *,
    weights: Union[str, Path],
    formats: Union[str, List[str]] = "onnx",
    imgsz: Union[int, Tuple[int, int]] = 640,
    device: Optional[str] = None,   # None lets Ultralytics choose; "cpu"|"cuda:0"|"mps"|"dla:0"
    half: bool = False,
    int8: bool = False,
    data: Optional[str] = None,     # representative dataset YAML for INT8
    fraction: float = 1.0,          # fraction of dataset for INT8 calibration
    dynamic: bool = False,
    batch: int = 1,
    opset: Optional[int] = None,
    simplify: bool = False,
    workspace: Optional[int] = None,  # GB for TensorRT builder
    nms: bool = False,
    out_dir: Union[str, Path] = "models/exports",
    run_name: Optional[str] = None,
    models_dir: Union[str, Path] = "models",
    download_models: bool = True,
) -> Dict[str, Union[str, List[str]]]:
    """
    Export an Ultralytics model to one or more formats and collect artifacts in run_dir.

    Returns:
      {
        "run_dir": str,
        "artifacts": [str, ...],   # moved artifacts' absolute paths
        "meta_path": str,          # export_meta.json
      }
    """
    if YOLO is None:  # pragma: no cover
        raise ImportError("ultralytics is required for export but failed to import") from _ULTRA_IMPORT_ERR

    formats_list = _norm_formats(formats)
    for f in formats_list:
        if f not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{f}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")

    # Resolve weights (path/URL/registry key)
    weights_path = resolve_weights_path(
        weights, models_dir=models_dir, detector=None, backend="ultralytics", allow_download=download_models
    )

    # Derive run dir
    stem = Path(weights_path).stem
    if run_name is None or not run_name:
        run_name = f"{stem}_export"
    run_dir = Path(out_dir) / run_name
    _ensure_out_dir(run_dir)

    model = YOLO(str(weights_path))

    exported_paths: List[str] = []
    export_args = dict(
        imgsz=imgsz,
        device=device,
        half=half,
        int8=int8,
        data=data,
        fraction=fraction,
        dynamic=dynamic,
        batch=batch,
        opset=opset,
        simplify=simplify,
        workspace=workspace,
        nms=nms,
        verbose=False,
    )

    # Export per-format and move artifacts into run_dir
    for fmt in formats_list:
        try:
            result = model.export(format=fmt, **export_args)
            results = result if isinstance(result, (list, tuple)) else [result]
        except Exception as e:
            raise RuntimeError(f"Export failed for format '{fmt}': {e}") from e

        for art in results:
            moved = _move_artifact(art, run_dir)
            if moved is not None:
                exported_paths.append(moved)

    # Write metadata
    meta = {
        "backend": "ultralytics",
        "weights": str(weights_path),
        "formats": formats_list,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in export_args.items()},
        "run_dir": str(run_dir.resolve()),
    }
    meta_path = run_dir / "export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir.resolve()),
        "artifacts": exported_paths,
        "meta_path": str(meta_path.resolve()),
    }


def list_models_ultralytics(*, models_dir: Union[str, Path] = "models") -> Dict[str, Dict]:
    """Helper for CLI: list registry + installed models (ultralytics-focused view)."""
    return {
        "registered": list_registered_models(detector=None, backend="ultralytics"),
        "installed": list_installed_models(models_dir=models_dir),
    }


__all__ = [
    "SUPPORTED_FORMATS",
    "export_model_ultralytics",
    "list_models_ultralytics",
]