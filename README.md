# detect

A modular **video object detection** toolkit with a clean **det-v1** JSON schema, pluggable backends, and optional model export.

Current backend:
- **Ultralytics YOLO** (bbox / pose / segmentation)

> By default, `detect` **does not write any files**. You opt-in to saving JSON, frames, or annotated video via flags.

---

## Features

- Detect videos → det-v1 JSON (always returned in-memory; optionally saved)
- Optional artifacts:
  - `--json` → save `detections.json`
  - `--frames` → save extracted frames
  - `--save-video <name.mp4>` → save annotated video
- YOLO tasks:
  - `yolo_bbox` (boxes)
  - `yolo_pose` (boxes + keypoints)
  - `yolo_seg` (boxes + polygons)
- Model registry keys:
  - pass `--weights yolo26n` / `yolo26n-seg` / `yolo26n-pose` (or a local path / URL)
- Exports:
  - export to formats like `onnx`, `engine`, `tflite`, `openvino`, `coreml`, etc (depending on toolchain)

---

## Install with `pip` (PyPI)

> Use this if you want to install and use the tool without cloning the repo.

### Install

```bash
pip install detect-lib
```

### Optional dependencies (pip extras)

Export helpers (ONNX + ONNXRuntime):

```bash
pip install "detect-lib[export]"
```

TensorFlow export paths (heavy):

```bash
pip install "detect-lib[tf]"
```

OpenVINO export:

```bash
pip install "detect-lib[openvino]"
```

CoreML export (macOS):

```bash
pip install "detect-lib[coreml]"
```

---

### CLI usage (pip)

Global help:

```bash
python -m detect.cli.detect_video -h
python -m detect.cli.export_model -h
```

List detectors:

```bash
python -c "import detect; print(detect.available_detectors())"
```

List models (registry + installed):

```bash
python -m detect.cli.detect_video --list-models
python -m detect.cli.export_model --list-models
```

Basic command (detection):

```bash
python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n
```

Basic command (export):

```bash
python -m detect.cli.export_model \
  --weights yolo26n \
  --formats onnx \
  --out-dir models/exports --run-name y26_onnx
```

---

## Python usage (import)

You can use `detect` as a library after installing `detect-lib` with pip.

### Quick sanity check

```bash
python -c "import detect; print(detect.available_detectors())"
```

### Run detection from a Python file

Create `run_detect.py`:

```python
from detect import detect_video

res = detect_video(
    video="in.mp4",
    detector="yolo_bbox",
    weights="yolo26n",
)

payload = res.payload
print(payload["schema_version"], len(payload["frames"]))
print(res.paths)  # populated only if you enable saving artifacts
```

Run:

```bash
python run_detect.py
```

### Run model export from a Python file

> Requires export extras based on the type of export needed (e.g., `pip install "detect-lib[export]"`).

Create `run_export.py`:

```python
from detect import export_model

res = export_model(
    weights="yolo26n",
    formats=["onnx"],
    imgsz=640,
    out_dir="models/exports",
    run_name="y26_onnx_py",
)

print("run_dir:", res["run_dir"])
print("artifacts:")
for p in res["artifacts"]:
    print(" -", p)
```

Run it:

```bash
python run_export.py
```

---

## Install from GitHub (uv)

Use this if you are developing locally or want reproducible project environments.

Install uv:  
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

```bash
uv --version
```

### Install dependencies

```bash
git clone https://github.com/Surya-Rayala/VideoPipeline-detection.git
cd VideoPipeline-detection
uv sync
```

### Optional dependencies (uv extras)

```bash
uv sync --extra export
uv sync --extra tf
uv sync --extra openvino
uv sync --extra coreml
```

---

## CLI usage (uv)

Global help:

```bash
uv run python -m detect.cli.detect_video -h
uv run python -m detect.cli.export_model -h
```

List detectors:

```bash
uv run python -c "import detect; print(detect.available_detectors())"
```

List models (registry + installed):

```bash
uv run python -m detect.cli.detect_video --list-models
uv run python -m detect.cli.export_model --list-models
```

Basic command (detection):

```bash
uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n
```

Basic command (export):

```bash
uv run python -m detect.cli.export_model \
  --weights yolo26n \
  --formats onnx \
  --out-dir models/exports --run-name y26_onnx
```

---

### TensorRT / engine export and run notes (important)

Exporting engine (TensorRT) typically requires an NVIDIA GPU + CUDA + TensorRT installed and version-compatible.

Export TensorRT engine:

```bash
uv run python -m detect.cli.export_model \
  --weights yolo26n \
  --formats engine \
  --device 0 \
  --out-dir models/exports --run-name y26_trt
```

Run / sanity-check the exported engine using this package (produces det-v1 output):

```bash
uv run python -m detect.cli.detect_video \
  --video in.mp4 \
  --detector yolo_bbox \
  --weights models/exports/y26_trt/yolo26n.engine \
  --device 0
```

Optionally save artifacts (JSON + annotated video):

```bash
uv run python -m detect.cli.detect_video \
  --video in.mp4 \
  --detector yolo_bbox \
  --weights models/exports/y26_trt/yolo26n.engine \
  --device 0 \
  --json \
  --save-video annotated_engine.mp4 \
  --out-dir out --run-name y26_trt_check
```

---

## CLI arguments

### Detection: `detect.cli.detect_video`

**Required**

- `--video <path>`: Path to the input video file.
- `--detector <name>`: Detector type (yolo_bbox, yolo_pose, or yolo_seg).
- `--weights <id|path|url>`: Registry key, local weights path, or URL to weights.

**Detection options**

- `--classes <ids>`: Filter to specific class IDs (comma/semicolon-separated), or omit for all classes.
- `--conf-thresh <float>`: Confidence threshold for detections (default 0.25).
- `--imgsz <int>`: Inference image size used by the backend (default 640).
- `--device <str>`: Device selector (e.g., `auto`, `cpu`, `mps`, `0`).
- `--half`: Enable FP16 inference where supported.

**Artifact saving (opt-in)**

- `--json`: Save detections.json under the run directory.
- `--frames`: Save extracted frames as images under the run directory.
- `--save-video <name.mp4>`: Save an annotated video under the run directory.
- `--display`: Show live visualization window while running (press q to quit).

**Output control**

- `--out-dir <dir>`: Output root directory used only if saving artifacts (default out).
- `--run-name <name>`: Run folder name inside out-dir (auto-derived if omitted).

**Model registry / downloads**

- `--models-dir <dir>`: Directory where models are stored/downloaded (default models).
- `--no-download`: Disable automatic download for registry keys/URLs.

**Misc**

- `--no-progress`: Disable progress bar output.
- `--list-models`: Print registry + installed models then exit.

---

### Export: `detect.cli.export_model`

**Required**

- `--weights <id|path|url>`: Registry key, local weights path, or URL to weights.

**Export options**

- `--formats <list>`: Comma/semicolon-separated export formats (default onnx).
- `--imgsz <int|H,W>`: Export image size as an int or `H,W` pair (default 640).
- `--device <str>`: Export device selector (e.g., `cpu`, `mps`, `0`).
- `--half`: Enable FP16 export where supported.
- `--int8`: Enable INT8 quantization (format/toolchain-dependent).
- `--data <yaml>`: Dataset YAML for INT8 calibration (when required).
- `--fraction <float>`: Fraction of dataset used for calibration (default 1.0).
- `--dynamic`: Enable dynamic shapes where supported.
- `--batch <int>`: Export batch size (default 1).
- `--opset <int>`: ONNX opset version (ONNX only).
- `--simplify`: Simplify the ONNX graph (ONNX only).
- `--workspace <int>`: TensorRT workspace size in GB (TensorRT only).
- `--nms`: Add NMS to exported model when supported by format/backend.

**Output control**

- `--out-dir <dir>`: Output root directory for exports (default models/exports).
- `--run-name <name>`: Export run folder name inside out-dir.

**Model registry / downloads**

- `--models-dir <dir>`: Directory where models are stored/downloaded (default models).
- `--no-download`: Disable automatic download for registry keys/URLs.

**Misc**

- `--list-models`: Print registry + installed models then exit.

---

## License

MIT License. See `LICENSE`.