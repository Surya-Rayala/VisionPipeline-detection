# detect

A modular **video detection** toolkit that produces a stable **det-v1** JSON output schema, with a pluggable backend (currently **Ultralytics**) and optional model export.

- Backend: **Ultralytics** (YOLO families, RT-DETR, YOLO-World/YOLOE, SAM/FastSAM — depending on your installed `ultralytics` version)
- Default behavior: **no files are written** unless you opt-in (JSON / frames / annotated video)

---

## Output schema (det-v1)

Every run returns a det-v1 payload in memory (and the CLI prints it to stdout).

Top-level keys:
- `schema_version`: always `"det-v1"`
- `video`: `{path, fps, frame_count, width, height}`
- `detector`: configuration used for the run (name/weights/conf/imgsz/device/half + task + optional prompts/topk)
- `frames`: list of per-frame records

Per-frame record:
- `frame`: 0-based frame index
- `file`: standard frame filename (e.g. `000000.jpg`) (even if frames aren’t saved)
- `detections`: list of detections

Detection fields:
- boxes: `bbox = [x1, y1, x2, y2]`
- pose: `keypoints = [[x, y, score], ...]`
- segmentation: `segments = [[[x, y], ...], ...]` (polygons)
- oriented boxes (best-effort): `obb = [cx, cy, w, h, angle_degrees]` plus an axis-aligned `bbox`

### Minimal example

```json
{
  "schema_version": "det-v1",
  "video": {"path": "in.mp4", "fps": 30.0, "frame_count": 120, "width": 1920, "height": 1080},
  "detector": {"name": "ultralytics", "weights": "yolo26n", "conf_thresh": 0.25, "imgsz": 640, "device": "cpu", "half": false, "task": "detect"},
  "frames": [
    {
      "frame": 0,
      "file": "000000.jpg",
      "detections": [
        {"det_ind": 0, "bbox": [100.0, 50.0, 320.0, 240.0], "score": 0.91, "class_id": 0, "class_name": "person"}
      ]
    }
  ]
}
```

---

## Install

Requires **Python 3.11+**.

### From PyPI

```bash
pip install detect-lib
```

Optional extras (only if you need them):

```bash
pip install "detect-lib[export]"      # ONNX / export helpers
pip install "detect-lib[coreml]"      # CoreML export (macOS)
pip install "detect-lib[openvino]"    # OpenVINO export
pip install "detect-lib[tf]"          # TensorFlow export paths (heavy)
```

### From GitHub (uv)

```bash
git clone https://github.com/Surya-Rayala/VisionPipeline-detection.git
cd VisionPipeline-detection
uv sync
```

Extras:

```bash
uv sync --extra export
uv sync --extra coreml
uv sync --extra openvino
uv sync --extra tf
```

---

## CLI

All CLI commands are:
- `python -m ...` (pip)
- `uv run python -m ...` (uv)

### Detection

Help:

```bash
python -m detect.cli.detect_video -h
```

List models (registry + installed):

```bash
python -m detect.cli.detect_video --list-models
```

#### Common patterns

**1) Bounding boxes (typical YOLO / RT-DETR)**

```bash
python -m detect.cli.detect_video \
  --video in.mp4 \
  --detector ultralytics \
  --weights yolo26n \
  --task detect \
  --json \
  --save-video annotated.mp4 \
  --out-dir out --run-name yolo26n_detect
```

**2) Instance segmentation (polygons)**

```bash
python -m detect.cli.detect_video \
  --video in.mp4 \
  --detector ultralytics \
  --weights yolo26n-seg \
  --task segment \
  --json \
  --save-video annotated.mp4 \
  --out-dir out --run-name yolo26n_seg
```

**3) Pose (keypoints)**

```bash
python -m detect.cli.detect_video \
  --video in.mp4 \
  --detector ultralytics \
  --weights yolo26n-pose \
  --task pose \
  --json \
  --save-video annotated.mp4 \
  --out-dir out --run-name yolo26n_pose
```

**4) Open-vocabulary (YOLO-World / YOLOE)**

```bash
python -m detect.cli.detect_video \
  --video in.mp4 \
  --detector ultralytics \
  --weights yolov8s-worldv2 \
  --task openvocab \
  --text "person,car,dog" \
  --json \
  --save-video annotated.mp4 \
  --out-dir out --run-name worldv2_openvocab
```

**Open-vocabulary + polygons (YOLOE *-seg)**

Use a YOLOE segmentation weight and `segment` when you want polygons.

```bash
python -m detect.cli.detect_video \
  --video in.mp4 \
  --detector ultralytics \
  --weights yoloe-11s-seg \
  --task segment \
  --text "person,car,dog" \
  --json \
  --save-video annotated.mp4 \
  --out-dir out --run-name yoloe_seg_openvocab
```

#### Task semantics (important)

- `detect | segment | pose | obb | classify | sam | sam2 | sam3 | fastsam` describe the **output type** you want.
- `openvocab` is a **prompt mode** for YOLO-World/YOLOE. Output type follows the model (boxes vs masks). If you want polygons, use a `*-seg` model and `segment`.

#### Prompts

You can supply prompts via:
- `--text "a,b,c"` (open-vocabulary label list)
- `--box "x1,y1,x2,y2"` (repeatable)
- `--point "x,y"` or `--point "x,y,label"` (repeatable; label 1=fg, 0=bg)
- `--prompts prompts.json` (combined)

Example `prompts.json`:

```json
{
  "text": ["person", "car", "dog"],
  "boxes": [[100, 100, 500, 500]],
  "points": [[320, 240, 1], [100, 120, 0]],
  "topk": 5
}
```

Export note (open-vocab): exported formats (ONNX/CoreML/etc.) may not support changing the vocabulary at runtime. If prompts don’t take effect, run the `.pt` weights for true open-vocabulary prompting or post-filter detections.

#### Artifacts (all opt-in)

- `--json` writes `out/<run-name>/detections.json`
- `--frames` writes `out/<run-name>/frames/*.jpg`
- `--save-video NAME.mp4` writes `out/<run-name>/NAME.mp4`

If you don’t enable any artifacts, no output directory is created.

---

## Python API

### Parameter mapping (Python vs CLI)

Python uses **snake_case** keyword arguments. The CLI uses **kebab-case** flags. The values are the same, but the names differ.

Common mapping:
- CLI `--video` → Python `video`
- CLI `--detector` → Python `detector`
- CLI `--weights` → Python `weights`
- CLI `--classes "0,2"` → Python `classes=[0, 2]`
- CLI `--conf-thresh` → Python `conf_thresh`
- CLI `--imgsz` → Python `imgsz`
- CLI `--device` → Python `device`
- CLI `--half` → Python `half=True`
- CLI `--task` → Python `task`

Prompts:
- CLI `--text "a,b"` → Python `prompts={"text": ["a", "b"]}`
- CLI `--box "x1,y1,x2,y2"` (repeatable) → Python `prompts={"boxes": [[x1, y1, x2, y2], ...]}`
- CLI `--point "x,y,label"` (repeatable) → Python `prompts={"points": [[x, y, label], ...]}`
- CLI `--topk N` → Python `topk=N` (or `prompts={"topk": N}`)

Artifacts (all opt-in):
- CLI `--json` → Python `save_json=True`
- CLI `--frames` → Python `save_frames=True`
- CLI `--save-video NAME.mp4` → Python `save_video="NAME.mp4"`
- CLI `--out-dir DIR` → Python `out_dir="DIR"`
- CLI `--run-name NAME` → Python `run_name="NAME"`
- CLI `--no-progress` → Python `progress=False`
- CLI `--display` → Python `display=True`

Note: the Python API also accepts an advanced `artifacts=ArtifactOptions(...)` object, but the convenience args above are easiest for most usage.

### Detect a video

```python
from detect import detect_video

res = detect_video(
    video="in.mp4",
    detector="ultralytics",
    weights="yolo26n",
    task="detect",
    classes=None,          # e.g. [0, 2] to filter class ids
    conf_thresh=0.25,
    imgsz=640,
    device="auto",
    half=False,
    # prompts={"text": ["person", "car", "dog"]},  # for open-vocabulary models
    save_json=True,
    save_video="annotated.mp4",
    out_dir="out",
    run_name="py_detect",
)

print(res.payload["schema_version"], len(res.payload["frames"]))
print(res.paths)
```

Note: legacy detector aliases (`yolo_bbox`, `yolo_seg`, `yolo_pose`) are still accepted for backward compatibility, but the docs use `ultralytics` everywhere.

---

## Export

Export is currently implemented for the Ultralytics backend.

### CLI export

```bash
python -m detect.cli.export_model -h

python -m detect.cli.export_model \
  --weights yolo26n \
  --formats onnx \
  --out-dir models/exports --run-name y26_onnx
```

### Export from Python

Python export also uses snake_case args (e.g., `out_dir`, `run_name`) and accepts `formats` as a list or comma-separated string.

```python
from detect.backends.ultralytics.export import export_model_ultralytics

res = export_model_ultralytics(
    weights="yolo26n",
    formats=["onnx"],
    imgsz=640,
    out_dir="models/exports",
    run_name="y26_onnx_py",
)

print("run_dir:", res["run_dir"])
for p in res["artifacts"]:
    print("-", p)
```

Compatibility notes:
- Some model families do not support export (e.g., MobileSAM and SAM/SAM2/SAM3 per Ultralytics docs). The export CLI will warn and exit.
- YOLO-World v1 weights (`*-world.pt`) do not support export; use YOLO-World v2 (`*-worldv2.pt`) for export.
- YOLOv10 supports export but only to a restricted set of formats; unsupported formats will warn and exit.

---

## License

MIT License. See `LICENSE`.
