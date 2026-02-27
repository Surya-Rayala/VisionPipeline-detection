from __future__ import annotations

"""
detect.cli.export_model
-----------------------

CLI wrapper for Ultralytics export (currently).

Example:
  python -m detect.cli.export_model \
    --weights yolo26n \
    --formats onnx,engine,tflite \
    --imgsz 640 \
    --int8 --data data/coco8.yaml --fraction 0.25 \
    --out-dir models/exports --run-name y26_export
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Union

from ..backends.ultralytics.export import (
    SUPPORTED_FORMATS,
    export_model_ultralytics,
    list_models_ultralytics,
)


def _parse_imgsz(s: str) -> Union[int, Tuple[int, int]]:
    if "," in s:
        a, b = s.split(",", 1)
        return (int(a.strip()), int(b.strip()))
    return int(s.strip())


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export Ultralytics YOLO models to multiple formats.")
    ap.add_argument("--weights", type=str, help="Weights path/URL/registry key (e.g., models/best.pt or yolo26n)")
    ap.add_argument("--formats", type=str, default="onnx", help=f"Comma/semicolon-separated formats (supported: {', '.join(sorted(SUPPORTED_FORMATS))})")
    ap.add_argument("--imgsz", type=str, default="640", help="Image size (int or H,W)")
    ap.add_argument("--device", type=str, default=None, help="Export device: auto|cpu|cuda:0|mps|dla:0 (None=auto)")
    ap.add_argument("--half", action="store_true", help="Enable FP16 export when supported")
    ap.add_argument("--int8", action="store_true", help="Enable INT8 quantization (often requires --data for calibration)")
    ap.add_argument("--data", type=str, default=None, help="Dataset YAML for INT8 calibration (e.g., data/coco8.yaml)")
    ap.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset for INT8 calibration")
    ap.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes where supported")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for export")
    ap.add_argument("--opset", type=int, default=None, help="ONNX opset")
    ap.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    ap.add_argument("--workspace", type=int, default=None, help="TensorRT workspace size (GB)")
    ap.add_argument("--nms", action="store_true", help="Add NMS where supported (e.g., TFLite/CoreML)")
    ap.add_argument("--out-dir", type=Path, default=Path("models/exports"))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--list-models", action="store_true", help="List registered and installed models, then exit")
    return ap


def run_cli() -> None:
    ap = _build_argparser()
    args = ap.parse_args()

    if args.list_models:
        info = list_models_ultralytics(models_dir=args.models_dir)
        print(json.dumps(info, indent=2))
        return

    if args.weights is None:
        ap.error("Missing required argument: --weights")

    try:
        res = export_model_ultralytics(
            weights=args.weights,
            formats=args.formats,
            imgsz=_parse_imgsz(args.imgsz),
            device=args.device,
            half=args.half,
            int8=args.int8,
            data=args.data,
            fraction=args.fraction,
            dynamic=args.dynamic,
            batch=args.batch,
            opset=args.opset,
            simplify=args.simplify,
            workspace=args.workspace,
            nms=args.nms,
            out_dir=args.out_dir,
            run_name=args.run_name,
            models_dir=args.models_dir,
            download_models=not args.no_download,
        )
    except SystemExit as e:
        # export.py uses SystemExit for user-facing capability errors
        msg = str(e)
        if msg:
            print(msg, file=sys.stderr)
        code = e.code if isinstance(e.code, int) else 2
        raise SystemExit(code)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    run_cli()


__all__ = ["run_cli"]