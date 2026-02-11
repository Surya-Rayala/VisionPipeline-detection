from __future__ import annotations

"""
detect.backends.ultralytics.detectors
------------------------------------

Ultralytics YOLO detector wrappers for:
- bounding boxes (yolo_bbox)
- pose (yolo_pose)
- segmentation (yolo_seg)

All implement BaseDetector.process_frame(frame_bgr) and return canonical Detection dicts
(without det_ind; runner assigns it).

This file intentionally mirrors your previous per-file detectors but consolidates them.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ...core.schema import BaseDetector, Detection

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore
    _ULTRALYTICS_IMPORT_ERROR = e

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _ultralytics_device_arg(device: str):
    """
    Convert our device string to what Ultralytics expects.

    Ultralytics expects:
      - "cpu", "mps", or GPU indices like "0" / "0,1"
    It does NOT accept "auto".
    """
    d = (device or "").strip().lower()

    if d in ("", "auto"):
        if torch is not None:
            # Prefer Apple MPS when available
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            # Prefer CUDA gpu index "0" when available
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "0"
        return "cpu"

    # Normalize common CUDA spellings
    if d == "cuda":
        return "0"
    if d.startswith("cuda:"):
        # "cuda:0" -> "0"
        return d.split(":", 1)[1].strip() or "0"

    # passthrough: "cpu", "mps", "0", "0,1", etc.
    return d

class _UltralyticsBase(BaseDetector):
    backend: str = "ultralytics"

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
        if YOLO is None:  # pragma: no cover
            raise ImportError(
                "ultralytics is required but failed to import"
            ) from _ULTRALYTICS_IMPORT_ERROR

        super().__init__(
            weights=weights,
            conf=conf,
            classes=classes,
            imgsz=imgsz,
            device=device,
            half=half,
        )
        self.model = YOLO(str(self.weights))

        # Class names map (list or dict depending on Ultralytics version)
        self.names = None
        try:
            self.names = getattr(self.model, "names", None) or getattr(self.model.model, "names", None)
        except Exception:
            self.names = None

    def _class_name(self, cls_id: int) -> Optional[str]:
        names = self.names
        if names is None:
            return None
        try:
            if isinstance(names, dict):
                return names.get(int(cls_id))
            if isinstance(names, (list, tuple)):
                if 0 <= int(cls_id) < len(names):
                    return str(names[int(cls_id)])
            return None
        except Exception:
            return None

    def _predict(self, frame_bgr: np.ndarray):
        # Ultralytics can take numpy BGR directly; it handles resize/letterbox internally.
        dev = _ultralytics_device_arg(self.device_str)
        return self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=dev,
            half=self.half,
            verbose=False,
        )

    def warmup(self) -> None:
        h = w = max(32, int(self.imgsz))
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        try:
            dev = _ultralytics_device_arg(self.device_str)
            _ = self.model.predict(
                source=dummy,
                conf=0.01,
                imgsz=self.imgsz,
                device=dev,
                half=self.half,
                verbose=False,
            )
        except Exception:
            pass


class YOLOBBoxDetector(_UltralyticsBase):
    """Ultralytics YOLO bounding-box detector."""

    # Keep this name stable for JSON readability
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._predict(frame_bgr)
        r = results[0]
        detections: List[Detection] = []

        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
        scores = r.boxes.conf.cpu().numpy().astype(float)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            score = float(scores[i])
            cid = int(cls_ids[i])
            det: Detection = {
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "class_id": cid,
            }
            cname = self._class_name(cid)
            if cname is not None:
                det["class_name"] = cname
            detections.append(det)
        return detections


class YOLOPoseDetector(_UltralyticsBase):
    """Ultralytics YOLO pose model wrapper (boxes + keypoints)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._predict(frame_bgr)
        r = results[0]
        detections: List[Detection] = []

        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
        scores = r.boxes.conf.cpu().numpy().astype(float)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        # Keypoints tensor: (N, K, 3) with (x, y, score)
        kpts = None
        if getattr(r, "keypoints", None) is not None:
            try:
                kpts = r.keypoints.data.cpu().numpy().astype(float)
            except Exception:
                kpts = None

        for i in range(boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            score = float(scores[i])
            cid = int(cls_ids[i])
            det: Detection = {
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "class_id": cid,
            }
            cname = self._class_name(cid)
            if cname is not None:
                det["class_name"] = cname

            if kpts is not None and i < kpts.shape[0]:
                kp_list = kpts[i].tolist()
                det["keypoints"] = [[float(a), float(b), float(c)] for a, b, c in kp_list]

            detections.append(det)

        return detections


class YOLOSegDetector(_UltralyticsBase):
    """Ultralytics YOLO segmentation model wrapper (boxes + polygon segments)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._predict(frame_bgr)
        r = results[0]
        detections: List[Detection] = []

        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
        scores = r.boxes.conf.cpu().numpy().astype(float)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        # r.masks.xy is typically list-like per instance
        mask_polys = None
        if getattr(r, "masks", None) is not None:
            try:
                mask_polys = r.masks.xy
            except Exception:
                mask_polys = None

        for i in range(boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            score = float(scores[i])
            cid = int(cls_ids[i])
            det: Detection = {
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "class_id": cid,
            }
            cname = self._class_name(cid)
            if cname is not None:
                det["class_name"] = cname

            segs: List[List[List[float]]] = []
            if mask_polys is not None and i < len(mask_polys):
                polys_i = mask_polys[i]
                # Ultralytics may return a list of polygons per instance
                if isinstance(polys_i, (list, tuple)):
                    for poly in polys_i:
                        if poly is None:
                            continue
                        arr = np.asarray(poly, dtype=float)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            segs.append(arr.tolist())
                else:
                    arr = np.asarray(polys_i, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        segs.append(arr.tolist())

            if segs:
                det["segments"] = segs

            detections.append(det)

        return detections


__all__ = [
    "YOLOBBoxDetector",
    "YOLOPoseDetector",
    "YOLOSegDetector",
]