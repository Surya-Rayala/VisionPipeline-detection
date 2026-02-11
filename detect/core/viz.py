from __future__ import annotations

"""
detect.core.viz
---------------

Visualization helpers for drawing detections on frames.

- Designed to be optional: only imported/used when display or save_video is enabled.
- Fixes segmentation visualization: `segments` is a list of polygons; we draw each polygon.
"""

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .schema import Detection

# Fixed light-green color (BGR), matching your original behavior
LIGHT_GREEN: Tuple[int, int, int] = (78, 238, 78)


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for visualization (cv2 import failed).")


def draw_keypoints(
    img: np.ndarray,
    keypoints: Optional[List[List[float]]],
    *,
    color: Tuple[int, int, int] = LIGHT_GREEN,
    radius: int = 4,
) -> None:
    """Draw keypoints as filled circles. keypoints: [[x,y,score], ...] or [[x,y], ...]."""
    _require_cv2()
    if not keypoints:
        return
    pts = np.asarray(keypoints, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return
    for row in pts:
        x, y = float(row[0]), float(row[1])
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)


def draw_polygon(
    img: np.ndarray,
    points: Optional[List[List[float]]],
    *,
    color: Tuple[int, int, int] = LIGHT_GREEN,
    alpha: float = 0.25,
    thickness: int = 2,
) -> None:
    """Draw a single polygon (list of [x,y]) with filled alpha overlay + outline."""
    _require_cv2()
    if not points:
        return
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        return

    poly = arr.astype(np.int32).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_detections(
    frame_bgr: np.ndarray,
    dets: List[Detection],
    frame_idx: int,
    *,
    color: Tuple[int, int, int] = LIGHT_GREEN,
) -> np.ndarray:
    """
    Draw boxes + optional keypoints + optional segments on a BGR frame.
    - `segments` is a list of polygons -> draws each polygon.
    """
    _require_cv2()
    out = frame_bgr.copy()

    for d in dets:
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        score = float(d.get("score", 0.0))
        label = d.get("class_name") or str(int(d.get("class_id", -1)))

        # Box + label
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"{label}:{score:.2f} f:{frame_idx}"
        cv2.putText(
            out,
            txt,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Keypoints: accept `keypoints` (canonical) or `kpts` (legacy)
        kps = d.get("keypoints") or d.get("kpts")
        if kps is not None:
            draw_keypoints(out, kps, color=color, radius=4)

        # Segments: canonical is list-of-polygons. Also accept `polygon` for legacy single polygon.
        segs = d.get("segments")
        if segs:
            # segs: List[polygon]; polygon: List[[x,y], ...]
            for poly in segs:
                draw_polygon(out, poly, color=color, alpha=0.25, thickness=2)
        else:
            poly = d.get("polygon")
            if poly is not None:
                draw_polygon(out, poly, color=color, alpha=0.25, thickness=2)

    return out


__all__ = [
    "LIGHT_GREEN",
    "draw_keypoints",
    "draw_polygon",
    "draw_detections",
]