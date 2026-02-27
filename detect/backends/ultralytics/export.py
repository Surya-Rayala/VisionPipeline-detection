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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ...registry.registry import resolve_weights_path, list_registered_models, list_installed_models

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore
    _ULTRA_IMPORT_ERR = e


SUPPORTED_FORMATS = {
    "torchscript", "onnx", "engine", "openvino", "coreml",
    "saved_model", "tflite", "edgetpu", "tfjs", "paddle",
    "mnn", "ncnn", "imx", "rknn", "pb",
}

# -----------------------------
# Export capability heuristics
# -----------------------------

# Some Ultralytics model families have export limitations (or no export at all).
# We keep this conservative and user-friendly: if we detect an incompatible model,
# we show a warning and exit with a compatible list.

# YOLOv10 has a narrower supported export format set (per Ultralytics docs).
_YOLOV10_EXPORT_FORMATS: Set[str] = {
    "torchscript",
    "onnx",
    "openvino",
    "engine",
    "coreml",
    "saved_model",
    "pb",
    "tflite",
    "edgetpu",
    "tfjs",
    "paddle",
}


def _infer_family(stem: str) -> str:
    s = stem.lower()
    if "world" in s:
        return "yolo_world"
    if "yoloe" in s:
        return "yoloe"
    if "sam2" in s or s.startswith("sam2"):
        return "sam2"
    if "sam3" in s or s.startswith("sam3"):
        return "sam3"
    if "sam" in s:
        # includes sam_b/sam_l/mobile_sam
        return "sam"
    if "fastsam" in s:
        return "fastsam"
    if "rtdetr" in s:
        return "rtdetr"
    if "yolov10" in s or s.startswith("yolov10"):
        return "yolov10"
    if "yolo_nas" in s or "yolonas" in s or "yolo-nas" in s:
        return "yolo_nas"
    if s.startswith("yolo") or "yolov" in s:
        return "yolo"
    return "unknown"


def _is_export_supported(stem: str) -> bool:
    s = stem.lower()

    # MobileSAM is inference-only in Ultralytics docs.
    if "mobile_sam" in s or "mobile-sam" in s:
        return False

    # SAM and SAM2 family models are inference-only in Ultralytics docs (Export ❌).
    # This includes sam_b/sam_l, sam2_*, sam2.* (sam2.1_*).
    if s.startswith("sam") or "sam2" in s or "sam3" in s:
        # Be conservative: block all SAM-family exports unless Ultralytics explicitly supports it.
        return False

    # YOLO-World v1 weights (e.g., yolov8s-world.pt) do not support export in Ultralytics docs.
    # v2 weights are typically named *-worldv2.pt.
    if "world" in s and "worldv2" not in s:
        return False

    return True


def _allowed_formats_for(stem: str) -> Optional[Set[str]]:
    """Return allowed formats for a given model stem, or None for default (no extra restriction)."""
    s = stem.lower()
    if "yolov10" in s or s.startswith("yolov10"):
        return set(_YOLOV10_EXPORT_FORMATS)
    return None


def _validate_export_capabilities(*, stem: str, formats_list: List[str]) -> None:
    if not _is_export_supported(stem):
        fam = _infer_family(stem)
        hint = ""
        if fam in {"sam", "sam2", "sam3"}:
            hint = "       Note: Ultralytics SAM/SAM2 model pages mark Export as unsupported (❌).\n"
        msg = (
            f"[warn] Export is not supported for this model family (detected: {fam}).\n"
            f"       Weights: {stem}\n"
            f"{hint}"
            "       Please choose an export-compatible model (e.g., YOLO-World v2 uses '*-worldv2.pt')."
        )
        raise SystemExit(msg)

    allowed = _allowed_formats_for(stem)
    if allowed is not None:
        bad = [f for f in formats_list if f not in allowed]
        if bad:
            msg = (
                f"[warn] Export format(s) not supported for this model (detected: {_infer_family(stem)}).\n"
                f"       Unsupported: {', '.join(bad)}\n"
                f"       Supported: {', '.join(sorted(allowed))}"
            )
            raise SystemExit(msg)


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

    # Capability checks (warn + exit for known-incompatible models/formats)
    # We evaluate after resolving weights so we can use the concrete filename.

    # Resolve weights (path/URL/registry key)
    weights_path = resolve_weights_path(
        weights, models_dir=models_dir, detector=None, backend="ultralytics", allow_download=download_models
    )

    stem = Path(weights_path).stem
    _validate_export_capabilities(stem=stem, formats_list=formats_list)

    # Derive run dir
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