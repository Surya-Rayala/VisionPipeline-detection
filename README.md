# detect

A modular **video object detection** toolkit with a clean **det-v1** JSON schema, pluggable backends, and optional model export.

Current backend:

- **Ultralytics YOLO** (bbox / pose / segmentation)

Future-friendly design:

- backend/plugin registry (`detect.backends`)
- model registry + weight resolver (`detect.registry`)
- stable det-v1 schema (`detect.core.schema`)
- optional exporters (today: Ultralytics export)

> By default, `detect` **does not write any files**. You opt-in to saving JSON, frames, or annotated video via flags.

---

## Features

- **Detect videos → det-v1 JSON** (returned in-memory, optionally saved)
- **Optional artifacts**
  - `--json` → save `detections.json`
  - `--frames` → save extracted frames
  - `--save-video <name.mp4>` → save annotated video
- **Supports YOLO tasks**
  - `yolo_bbox` (boxes)
  - `yolo_pose` (boxes + keypoints)
  - `yolo_seg` (boxes + polygons)
- **Model registry keys**
  - pass `--weights yolo26n` / `yolo26n-seg` / etc (or a local path / URL)
- **Model export**
  - export to formats like `onnx`, `engine`, `tflite`, etc (depending on platform/toolchain)

---

## Recommended environment (Python 3.11)

This project targets **Python 3.11+**.

Why:

- avoids wheel availability issues for exporters (e.g. `onnxruntime`)
- works well with modern `torch` / `ultralytics`

---

## Install from GitHub (uv)

Install `uv` using Astral’s installer:

https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

```bash
uv --version
```

Clone + install base deps:

```bash
git clone <YOUR_REPO_URL>.git
cd detect
uv sync
```

> `uv sync` installs only the base dependencies (detection). Export extras are opt-in.

---

## Optional dependencies (exports & runtimes)

Your `pyproject.toml` defines extras:

- `export`: ONNX + ONNXRuntime
- `tf`: TensorFlow export paths (heavy)
- `openvino`: OpenVINO export
- `coreml`: CoreML export (macOS)

Install extras like:

```bash
uv sync --extra export
uv sync --extra tf
uv sync --extra openvino
uv sync --extra coreml
```

You can combine them:

```bash
uv sync --extra export --extra openvino
```

### Torch notes (CPU / CUDA / Apple Silicon)

Ultralytics uses **PyTorch**. On some platforms you may want to install torch separately depending on acceleration:

- **Apple Silicon (MPS)**: regular pip/uv torch installs usually work.
- **CUDA**: install torch using the official PyTorch selector for your CUDA version, then run `uv sync` for the rest.

---

## CLI usage (uv)

Run CLIs using `uv run` so you’re always using the project environment:

### Global help

```bash
uv run python -m detect.cli.detect_video -h
uv run python -m detect.cli.export_model -h
```

### List detectors

```bash
uv run python -c "import detect; print(detect.available_detectors())"
```

### List models (registry + installed)

```bash
uv run python -m detect.cli.detect_video --list-models
uv run python -m detect.cli.export_model --list-models
```

---

## Detection CLI

Module:

```bash
uv run python -m detect.cli.detect_video ...
```

### Arguments

**Required**

- `--video <path>`: input video path
- `--detector <name>`: one of `yolo_bbox | yolo_pose | yolo_seg`
- `--weights <id>`: weights identifier
  - registry key (e.g. `yolo26n`, `yolo26n-seg`, `yolo26n-pose`)
  - OR local path (e.g. `models/yolo26n.pt`)
  - OR URL (downloaded into `models/` if downloads enabled)

**Common optional**

- `--classes <ids>`: comma/semicolon-separated class ids (e.g. `"0,2"`). If omitted, all classes.
- `--conf-thresh <float>`: confidence threshold (default: `0.25`)
- `--imgsz <int>`: inference image size (default: `640`)
- `--device <str>`: device selection
  - `auto` (default; mapped to `mps`/GPU/`cpu` depending on availability)
  - `cpu`, `mps`, `0`, `0,1`, etc.
- `--half`: enable FP16 inference where supported

**Artifacts (opt-in)**

- `--json`: save `detections.json`
- `--frames`: save extracted frames as images
- `--save-video <name.mp4>`: save annotated video under the run directory
- `--display`: show a live window (press `q` to quit)

**Output control**

- `--out-dir <dir>`: root output directory (only used if saving artifacts; default: `out`)
- `--run-name <name>`: run folder name inside `out-dir` (if omitted, derived from video + detector)

**Model registry / downloads**

- `--models-dir <dir>`: where models are stored/downloaded (default: `models`)
- `--no-download`: disable automatic download for registry keys/URLs

**Misc**

- `--no-progress`: disable tqdm progress bar
- `--list-models`: print registry + installed models then exit

### Examples

#### In-memory only (default; saves nothing)

```bash
uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n
```

- Prints det-v1 JSON to stdout
- Does **not** create `out/` unless you save artifacts

#### Save JSON only

```bash
uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n \
  --json \
  --out-dir out --run-name run_json
```

#### Save JSON + frames

```bash
uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n \
  --json --frames \
  --out-dir out --run-name run_frames
```

#### Save annotated video (segmentation example)

```bash
uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_seg \
  --weights yolo26n-seg \
  --save-video annotated.mp4 \
  --out-dir out --run-name run_seg
```

#### Live display (press `q` to quit)

```bash
uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_pose \
  --weights yolo26n-pose \
  --display
```

---

## det-v1 JSON schema (stable)

The tool outputs a canonical JSON schema (`schema_version: "det-v1"`):

- `video`: source metadata (fps, size, frame_count, path)
- `detector`: settings (name, backend, weights, conf/imgsz/device, etc.)
- `frames`: list of per-frame detections
  - bbox: `[x1, y1, x2, y2]`
  - pose: `keypoints = [[x, y, score], ...]`
  - seg: `segments = list of polygons [[[x, y], ...], ...]`

This schema is defined in `detect/core/schema.py`.

---

## Python usage (import)

### Quick sanity check

```bash
uv run python -c "import detect; print(detect.available_detectors())"
```

### Run detection in code (returns payload + paths)

Create `run_detect.py`:

```python
from detect.core.run import detect_video
from detect.core.artifacts import ArtifactOptions

res = detect_video(
    video="in.mp4",
    detector="yolo_bbox",
    weights="yolo26n",
    artifacts=ArtifactOptions(
        save_json=False,
        save_frames=False,
        save_video=False,
    ),
)

# det-v1 payload always available in memory:
payload = res.payload
print(payload["schema_version"], len(payload["frames"]))

# If you enabled saving, paths would show up here:
print(res.paths)
```

Run:

```bash
uv run python run_detect.py
```

---

## Export CLI

Module:

```bash
uv run python -m detect.cli.export_model ...
```

> Export support depends on format + your platform/toolchain. Start with **ONNX**.

### Install export extras

```bash
uv sync --extra export
```

### Arguments

**Required**

- `--weights <id>`: weights path/URL/registry key (e.g. `yolo26n`)

**Common optional**

- `--formats <list>`: comma/semicolon-separated formats (default: `onnx`)
  - examples: `onnx`, `engine`, `tflite`, `openvino`, `coreml`, ...
- `--imgsz <int | H,W>`: export image size (default: `640`)
- `--device <str>`: export device (commonly `cpu`, `mps`, `0`)
- `--half`: enable FP16 export where supported
- `--int8`: enable INT8 export (usually requires `--data`)
- `--data <yaml>`: dataset YAML for INT8 calibration (format-dependent)
- `--fraction <float>`: fraction of dataset used for calibration (default: `1.0`)
- `--dynamic`: enable dynamic shapes where supported
- `--batch <int>`: export batch size (default: `1`)
- `--opset <int>`: ONNX opset version
- `--simplify`: simplify ONNX graph
- `--workspace <int>`: TensorRT workspace (GB)
- `--nms`: add NMS in exported model (format-dependent)

**Output control**

- `--out-dir <dir>`: output root for exports (default: `models/exports`)
- `--run-name <name>`: folder name under out-dir (default derived from weights)

**Model registry / downloads**

- `--models-dir <dir>`: where models are stored/downloaded (default: `models`)
- `--no-download`: disable automatic download for registry keys/URLs

**Misc**

- `--list-models`: print registry + installed models then exit

### Examples

#### Export ONNX

```bash
uv run python -m detect.cli.export_model \
  --weights yolo26n \
  --formats onnx \
  --out-dir models/exports --run-name y26_onnx
```

#### Validate ONNX + run a minimal ONNXRuntime session

```bash
# 1) Check file integrity
uv run python -c "import onnx; m=onnx.load('models/exports/y26_onnx/yolo26n.onnx'); onnx.checker.check_model(m); print('ONNX OK')"

# 2) Print ORT version
uv run python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"
```

> Running full ONNX inference requires knowing the model’s input/output names and preprocessing. For quick sanity checks, the `onnx.checker` validation above is usually sufficient.

---

## TensorRT / `engine` export and run notes (important)

Exporting `engine` (TensorRT) typically requires:

- NVIDIA GPU
- CUDA toolkit compatible with your GPU driver
- TensorRT installed (often via NVIDIA packages, not pure pip)
- matching versions across torch / CUDA / TensorRT

### Export TensorRT engine

```bash
uv run python -m detect.cli.export_model \
  --weights yolo26n \
  --formats engine \
  --device 0 \
  --out-dir models/exports --run-name y26_trt
```

### Run / sanity-check the exported engine

The easiest way to sanity-check a `.engine` artifact is to run Ultralytics predict using it:

```bash
# Example: run the engine on a video
uv run python -c "from ultralytics import YOLO; m=YOLO('models/exports/y26_trt/yolo26n.engine'); r=m.predict(source='in.mp4', device=0, verbose=False); print('OK', len(r))"
```

If this fails, it’s usually an environment/toolchain issue, not the repo code.

---

## Troubleshooting

### 1) `Can't get attribute 'Segment26' ...`

This means **weights are newer than your installed `ultralytics`**. Upgrade:

```bash
uv add -U ultralytics
uv sync
```

### 2) Export wheels missing for your Python

Some exporter runtimes (e.g. `onnxruntime`) may not ship wheels for older Python versions.
This project uses **Python 3.11+** to avoid that.

---

## License

MIT License. See `LICENSE`.