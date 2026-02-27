from __future__ import annotations

"""
detect.backends.ultralytics.detectors
------------------------------------

Ultralytics detector wrappers.

This module provides:
- A universal detector `UltralyticsDetector` that can normalize outputs for multiple
  Ultralytics model families/tasks using a `task` switch (default: "auto").

All implement BaseDetector.process_frame(frame_bgr) and return canonical Detection dicts
(without det_ind; runner assigns it).

Supported tasks (best-effort, depends on installed ultralytics version/models):
- detect (boxes)
- segment (boxes + masks -> polygons)
- pose (boxes + keypoints)
- obb (oriented boxes; stored in Detection.obb + bbox fallback)
- classify (frame-level classification mapped to a full-frame bbox)
- openvocab (YOLO-World/YOLOE style; uses prompts['text'] via model.set_classes when supported)
- sam / sam2 / sam3 / fastsam (promptable segmentation; best-effort prompt passing)

Notes:
- Prompt handling across SAM/SAM2/FastSAM is intentionally defensive because APIs vary by version.
- Prompts mapping:
  - openvocab uses prompts['text']; SAM/FastSAM uses prompts['boxes'] and/or prompts['points'] (labels derived from points).
"""

from pathlib import Path
import os
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

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


logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    """Return True if Ultralytics debug logging is enabled.

    Enable by setting DETECT_ULTRA_DEBUG=1 (or true/yes/on).
    """
    v = (os.getenv("DETECT_ULTRA_DEBUG", "") or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _ultralytics_device_arg(device: str) -> str:
    """Convert our device string to what Ultralytics expects."""
    d = (device or "").strip().lower()

    if d in ("", "auto"):
        if torch is not None:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "0"
        return "cpu"

    if d == "cuda":
        return "0"
    if d.startswith("cuda:"):
        return d.split(":", 1)[1].strip() or "0"

    return d


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    """Best-effort convert torch/np/list-ish to numpy array."""
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def _bbox_from_poly(poly_xy: Sequence[Sequence[float]]) -> Optional[List[float]]:
    try:
        arr = np.asarray(poly_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            return None
        x1 = float(np.min(arr[:, 0]))
        y1 = float(np.min(arr[:, 1]))
        x2 = float(np.max(arr[:, 0]))
        y2 = float(np.max(arr[:, 1]))
        return [x1, y1, x2, y2]
    except Exception:
        return None


def _safe_call_predict(model: Any, *, source: np.ndarray, base_kwargs: Dict[str, Any], extra_kwargs: Dict[str, Any]) -> Any:
    """Call model.predict defensively: drop unsupported kwargs on TypeError."""
    kwargs = dict(base_kwargs)
    kwargs.update(extra_kwargs)

    if not hasattr(model, "predict"):
        # Some Ultralytics objects are callable
        return model(source, **kwargs)

    # First try full kwargs
    try:
        return model.predict(source=source, **kwargs)
    except TypeError as e:
        first_err = e

    # Drop extra keys one-by-one (best effort)
    drop_keys = list(extra_kwargs.keys())
    for k in drop_keys:
        try_kwargs = dict(base_kwargs)
        for kk, vv in extra_kwargs.items():
            if kk != k:
                try_kwargs[kk] = vv
        try:
            out = model.predict(source=source, **try_kwargs)
            if _debug_enabled():
                logger.warning(
                    "Ultralytics predict() rejected kwarg '%s'; retry succeeded after dropping it.",
                    k,
                )
            return out
        except TypeError:
            continue

    # Last resort: base kwargs only
    if _debug_enabled() and extra_kwargs:
        logger.warning(
            "Ultralytics predict() rejected extra kwargs %s; falling back to base kwargs only.",
            sorted(extra_kwargs.keys()),
        )
        logger.debug("First TypeError from predict() with full kwargs: %s", first_err)

    return model.predict(source=source, **base_kwargs)


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
        # New:
        task: str = "auto",
        prompts: Optional[Dict[str, Any]] = None,
        topk: Optional[int] = None,
    ) -> None:
        if YOLO is None:  # pragma: no cover
            raise ImportError("ultralytics is required but failed to import") from _ULTRALYTICS_IMPORT_ERROR

        super().__init__(
            weights=weights,
            conf=conf,
            classes=classes,
            imgsz=imgsz,
            device=device,
            half=half,
        )

        self.task = (task or "auto").strip().lower()
        self.prompts: Dict[str, Any] = dict(prompts or {})
        self.topk = int(topk) if topk is not None else None

        # Default model loader uses YOLO. For SAM/FastSAM we try to load specialized classes if present.
        self.model = self._load_model(str(self.weights), task=self.task)

        # Class names map (list or dict depending on Ultralytics version)
        self.names = None
        try:
            self.names = getattr(self.model, "names", None) or getattr(getattr(self.model, "model", None), "names", None)
        except Exception:
            self.names = None

        # Open-vocabulary models may support set_classes([...])
        txt = self.prompts.get("text")
        if txt and hasattr(self.model, "set_classes"):
            try:
                # Accept either list[str] or comma-separated string
                if isinstance(txt, str):
                    classes_txt = [t.strip() for t in txt.replace(";", ",").split(",") if t.strip()]
                else:
                    classes_txt = [str(t).strip() for t in txt if str(t).strip()]
                if classes_txt:
                    self.model.set_classes(classes_txt)
            except Exception:
                pass

    def _load_model(self, weights: str, *, task: str) -> Any:
        """Load a model instance.

        Rules:
        - SAM-family weights (sam_*, sam2_*, sam2.*_*, sam3*) must be loaded with `ultralytics.SAM`.
          Falling back to `YOLO(weights)` will crash because checkpoint structure differs.
        - FastSAM weights should be loaded with `ultralytics.FastSAM` when available.
        - Everything else uses `ultralytics.YOLO`.
        """
        t = (task or "auto").lower().strip()
        wstem = Path(weights).stem.lower()

        # Determine if this is SAM-family (task override or filename)
        is_sam = t in {"sam", "sam2", "sam3"} or wstem.startswith("sam") or "sam2" in wstem or "mobile_sam" in wstem

        if is_sam:
            try:
                from ultralytics import SAM  # type: ignore

                return SAM(weights)
            except Exception as e:
                raise ImportError(
                    "Failed to load SAM/SAM2 weights via `from ultralytics import SAM`. "
                    "Please ensure you have a recent `ultralytics` installed that includes SAM support."
                ) from e

        if t == "fastsam" or "fastsam" in wstem:
            try:
                from ultralytics import FastSAM  # type: ignore

                return FastSAM(weights)
            except Exception:
                # If FastSAM class isn't available, YOLO can sometimes load FastSAM weights.
                # We'll allow fallback here.
                pass

        return YOLO(weights)

    def _class_name(self, cls_id: int) -> Optional[str]:
        names = self.names
        if names is None:
            return None
        try:
            if isinstance(names, dict):
                v = names.get(int(cls_id))
                return None if v is None else str(v)
            if isinstance(names, (list, tuple)):
                if 0 <= int(cls_id) < len(names):
                    return str(names[int(cls_id)])
            return None
        except Exception:
            return None

    def _predict(self, frame_bgr: np.ndarray) -> Any:
        dev = _ultralytics_device_arg(self.device_str)

        base_kwargs: Dict[str, Any] = dict(
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=dev,
            half=self.half,
            verbose=False,
        )

        # Prompt fields (best-effort) used by promptable segmentation tasks (SAM/SAM2/FastSAM).
        extra_kwargs: Dict[str, Any] = {}

        # Only SAM/SAM2/SAM3/FastSAM style models accept prompt kwargs (bboxes/points/labels/texts).
        # For YOLO-World/YOLOE we set classes via model.set_classes(...) during __init__, and then
        # call predict() with standard args only.
        promptable = self.task in {"sam", "sam2", "sam3", "fastsam"}
        if promptable and self.prompts:
            if "boxes" in self.prompts and self.prompts["boxes"] is not None:
                extra_kwargs["bboxes"] = self.prompts["boxes"]

            if "points" in self.prompts and self.prompts["points"] is not None:
                pts = self.prompts["points"]
                extra_kwargs["points"] = [[float(p[0]), float(p[1])] for p in pts]
                labels: List[int] = []
                for p in pts:
                    if len(p) >= 3:
                        labels.append(int(p[2]))
                    else:
                        labels.append(1)
                extra_kwargs["labels"] = labels

            if "text" in self.prompts and self.prompts["text"] is not None:
                # Some promptable models accept 'texts' (plural); use that key to avoid collisions with viz args like 'text'.
                extra_kwargs["texts"] = self.prompts["text"]

        return _safe_call_predict(self.model, source=frame_bgr, base_kwargs=base_kwargs, extra_kwargs=extra_kwargs)

    def warmup(self) -> None:
        h = w = max(32, int(self.imgsz))
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        try:
            _ = self._predict(dummy)
        except Exception:
            pass


class UltralyticsDetector(_UltralyticsBase):
    """Universal Ultralytics detector controlled by `task` (default: auto)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _infer_task(self, r: Any) -> str:
        # Explicit task override
        if self.task and self.task != "auto":
            return self.task

        # Model-reported task
        try:
            mt = getattr(self.model, "task", None)
            if isinstance(mt, str) and mt:
                return mt.lower()
        except Exception:
            pass

        # Heuristic from result fields
        if getattr(r, "probs", None) is not None:
            return "classify"
        if getattr(r, "obb", None) is not None:
            return "obb"
        if getattr(r, "keypoints", None) is not None:
            return "pose"
        if getattr(r, "masks", None) is not None:
            return "segment"
        if getattr(r, "boxes", None) is not None:
            return "detect"
        return "detect"

    def process_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._predict(frame_bgr)
        # Ultralytics typically returns a list-like of results
        r = results[0] if isinstance(results, (list, tuple)) and results else results

        task = self._infer_task(r)

        # Open-vocabulary is a prompt mode. Route to segmentation when masks are present,
        # otherwise fall back to standard bbox detection.
        if task in {"openvocab", "open-vocab", "open_vocab"}:
            if getattr(r, "masks", None) is not None:
                return self._process_segment(r)
            return self._process_detect(r)

        # Classification
        if task in {"classify", "cls", "class"}:
            return self._process_classify(frame_bgr, r)

        # OBB
        if task in {"obb"}:
            return self._process_obb(r)

        # Pose
        if task in {"pose"}:
            return self._process_pose(r)

        # Segmentation / promptable segmentation
        if task in {"segment", "seg", "sam", "sam2", "sam3", "fastsam"}:
            return self._process_segment(r)

        # Default detection
        return self._process_detect(r)

    def _process_detect(self, r: Any) -> List[Detection]:
        detections: List[Detection] = []
        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = _to_numpy(r.boxes.xyxy)
        scores = _to_numpy(r.boxes.conf)
        cls_ids = _to_numpy(r.boxes.cls)
        if boxes_xyxy is None or scores is None or cls_ids is None:
            return detections

        boxes_xyxy = boxes_xyxy.astype(float)
        scores = scores.astype(float)
        cls_ids = cls_ids.astype(int)

        # Open-vocabulary prompt labels (if provided). Prefer these for class_name
        # because some models (e.g. YOLO-World) may keep COCO names internally.
        prompt_labels: Optional[List[str]] = None
        txt = self.prompts.get("text")
        if isinstance(txt, str) and txt.strip():
            prompt_labels = [t.strip() for t in txt.replace(";", ",").split(",") if t.strip()]
        elif isinstance(txt, list) and txt:
            prompt_labels = [str(t).strip() for t in txt if str(t).strip()]
        if prompt_labels is not None and not prompt_labels:
            prompt_labels = None

        for i in range(int(boxes_xyxy.shape[0])):
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i].tolist()]
            score = float(scores[i])
            cid = int(cls_ids[i])
            det: Detection = {"bbox": [x1, y1, x2, y2], "score": score, "class_id": cid}
            # Prefer prompt label names when available (open-vocabulary).
            if prompt_labels is not None and 0 <= cid < len(prompt_labels):
                det["class_name"] = prompt_labels[cid]
            else:
                cname = self._class_name(cid)
                if cname is not None:
                    det["class_name"] = cname

            # Attach the prompt list used for this run (compact string) for traceability.
            if prompt_labels is not None:
                det["text_prompt"] = ",".join(prompt_labels)
            detections.append(det)
        return detections

    def _process_pose(self, r: Any) -> List[Detection]:
        detections = self._process_detect(r)
        if not detections:
            return detections

        kpts = None
        if getattr(r, "keypoints", None) is not None:
            try:
                kpts = _to_numpy(r.keypoints.data)
                if kpts is not None:
                    kpts = kpts.astype(float)
            except Exception:
                kpts = None

        if kpts is None:
            return detections

        for i, det in enumerate(detections):
            if i < kpts.shape[0]:
                kp_list = kpts[i].tolist()
                det["keypoints"] = [[float(a), float(b), float(c)] for a, b, c in kp_list]
        return detections

    def _process_segment(self, r: Any) -> List[Detection]:
        detections: List[Detection] = []

        # Try to start from boxes if present
        has_boxes = getattr(r, "boxes", None) is not None and len(getattr(r, "boxes", [])) > 0
        if has_boxes:
            detections = self._process_detect(r)
        else:
            detections = []

        # Masks/polygons
        mask_polys = None
        if getattr(r, "masks", None) is not None:
            try:
                mask_polys = r.masks.xy
            except Exception:
                mask_polys = None

        # Best-effort mask scores (SAM/FastSAM). Different Ultralytics versions may expose
        # these as `masks.conf` or `masks.scores`.
        mask_scores = None
        if getattr(r, "masks", None) is not None:
            try:
                ms = getattr(r.masks, "conf", None)
                if ms is None:
                    ms = getattr(r.masks, "scores", None)
                mask_scores = _to_numpy(ms)
                if mask_scores is not None:
                    mask_scores = mask_scores.astype(float).reshape(-1)
            except Exception:
                mask_scores = None

        if mask_polys is None:
            # No masks -> return box detections (or empty)
            return detections

        # Attach polygons. If no boxes existed, create detections per polygon.
        if not detections:
            for i in range(len(mask_polys)):
                polys_i = mask_polys[i]
                segs: List[List[List[float]]] = []
                if isinstance(polys_i, (list, tuple)):
                    for poly in polys_i:
                        arr = np.asarray(poly, dtype=float)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            segs.append(arr.tolist())
                else:
                    arr = np.asarray(polys_i, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        segs.append(arr.tolist())

                if not segs:
                    continue
                # Make a best-effort bbox from first polygon
                bb = _bbox_from_poly(segs[0]) or [0.0, 0.0, 0.0, 0.0]
                score = 1.0
                if mask_scores is not None and i < mask_scores.size:
                    score = float(mask_scores[i])
                    if score < float(self.conf):
                        continue

                det: Detection = {"bbox": bb, "score": float(score), "class_id": -1, "segments": segs}
                detections.append(det)
            return detections

        # We have detections and polygons; align by index
        for i, det in enumerate(detections):
            if i >= len(mask_polys):
                break
            polys_i = mask_polys[i]
            segs: List[List[List[float]]] = []
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
        return detections

    def _process_obb(self, r: Any) -> List[Detection]:
        detections: List[Detection] = []

        obb = getattr(r, "obb", None)
        if obb is None:
            # Fallback: try regular detect
            return self._process_detect(r)

        # Ultralytics OBB outputs vary; attempt to read:
        # - obb.xywhr (cx,cy,w,h,angle) and obb.conf/obb.cls
        # - or obb.xyxyxyxy (8 coords) and obb.conf/obb.cls
        xywhr = None
        xyxyxyxy = None
        conf = None
        cls = None

        try:
            xywhr = _to_numpy(getattr(obb, "xywhr", None))
        except Exception:
            xywhr = None
        try:
            xyxyxyxy = _to_numpy(getattr(obb, "xyxyxyxy", None))
        except Exception:
            xyxyxyxy = None
        try:
            conf = _to_numpy(getattr(obb, "conf", None))
        except Exception:
            conf = None
        try:
            cls = _to_numpy(getattr(obb, "cls", None))
        except Exception:
            cls = None

        if conf is None and getattr(r, "boxes", None) is not None:
            try:
                conf = _to_numpy(r.boxes.conf)
            except Exception:
                conf = None
        if cls is None and getattr(r, "boxes", None) is not None:
            try:
                cls = _to_numpy(r.boxes.cls)
            except Exception:
                cls = None

        if conf is not None:
            conf = conf.astype(float)
        if cls is not None:
            cls = cls.astype(int)

        if xywhr is not None:
            xywhr = xywhr.astype(float)
            for i in range(int(xywhr.shape[0])):
                cx, cy, w, h, ang = [float(v) for v in xywhr[i].tolist()[:5]]
                score = float(conf[i]) if conf is not None and i < len(conf) else 1.0
                cid = int(cls[i]) if cls is not None and i < len(cls) else -1
                # bbox fallback from center/size (axis-aligned)
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
                det: Detection = {"bbox": [x1, y1, x2, y2], "score": score, "class_id": cid, "obb": [cx, cy, w, h, ang]}
                cname = self._class_name(cid)
                if cname is not None:
                    det["class_name"] = cname
                detections.append(det)
            return detections

        if xyxyxyxy is not None:
            xyxyxyxy = xyxyxyxy.astype(float)
            for i in range(int(xyxyxyxy.shape[0])):
                pts = xyxyxyxy[i].reshape(-1, 2)
                bb = _bbox_from_poly(pts.tolist()) or [0.0, 0.0, 0.0, 0.0]
                score = float(conf[i]) if conf is not None and i < len(conf) else 1.0
                cid = int(cls[i]) if cls is not None and i < len(cls) else -1
                det: Detection = {"bbox": bb, "score": score, "class_id": cid}
                # No canonical corners field in det-v1; store a coarse xywhr-like obb if possible
                det["obb"] = []  # placeholder; downstream can treat presence of obb as OBB
                cname = self._class_name(cid)
                if cname is not None:
                    det["class_name"] = cname
                detections.append(det)
            return detections

        # Fallback
        return self._process_detect(r)

    def _process_classify(self, frame_bgr: np.ndarray, r: Any) -> List[Detection]:
        probs = getattr(r, "probs", None)
        if probs is None:
            return []
        data = getattr(probs, "data", None)
        src = data if data is not None else probs
        p = _to_numpy(src)
        if p is None:
            return []
        p = p.astype(float).reshape(-1)

        # Determine top-k
        k = self.topk
        if k is None:
            k = int(self.prompts.get("topk") or 1)
        k = max(1, int(k))

        # Argmax class
        cid = int(np.argmax(p)) if p.size else -1
        score = float(p[cid]) if cid >= 0 and cid < p.size else 0.0

        # Full-frame bbox
        h, w = frame_bgr.shape[:2]
        det: Detection = {
            "bbox": [0.0, 0.0, float(w), float(h)],
            "score": score,
            "class_id": cid,
        }
        cname = self._class_name(cid)
        if cname is not None:
            det["class_name"] = cname

        # Store top-k probabilities (optional)
        try:
            top_idx = np.argsort(-p)[:k]
            det["probs"] = [float(p[i]) for i in top_idx]
        except Exception:
            det["probs"] = None  # type: ignore

        return [det]


__all__ = [
    "UltralyticsDetector",
]