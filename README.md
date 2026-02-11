# detect

A modular **video object detection** toolkit with a clean **det-v1** JSON schema, pluggable backends, and optional model export.

Current backend:
- **Ultralytics YOLO** (bbox / pose / segmentation)

Architecture highlights:
- backend/plugin registry (`detect.backends`)
- model registry + weight resolver (`detect.registry`)
- stable det-v1 schema (`detect.core.schema`)
- optional exporters (today: Ultralytics export)

> By default, `detect` **does not write any files**. You opt-in to saving JSON, frames, or annotated video via flags.

---

## Features

- **Detect videos → det-v1 JSON** (always returned in-memory; optionally saved)
- **Optional artifacts**
  - `--json` → save `detections.json`
  - `--frames` → save extracted frames
  - `--save-video <name.mp4>` → save annotated video
- **YOLO tasks**
  - `yolo_bbox` (boxes)
  - `yolo_pose` (boxes + keypoints)
  - `yolo_seg` (boxes + polygons)
- **Model registry keys**
  - pass `--weights yolo26n` / `yolo26n-seg` / `yolo26n-pose` (or a local path / URL)
- **Exports**
  - export to formats like `onnx`, `engine`, `tflite`, `openvino`, `coreml`, etc (depending on toolchain)

---

## Recommended environment (Python 3.11+)

This project targets **Python 3.11+**.

---

## Install with `pip` (PyPI)

> Use this if you want to install and use the tool without cloning the repo.

### Install (base detection)

```bash
pip install detect

Optional installs (pip extras)

Export helpers (ONNX + ONNXRuntime):

pip install "detect[export]"

TensorFlow export paths (heavy):

pip install "detect[tf]"

OpenVINO export:

pip install "detect[openvino]"

CoreML export (macOS):

pip install "detect[coreml]"


⸻

CLI usage (pip)

When installed with pip, run modules directly:

Global help

python -m detect.cli.detect_video -h
python -m detect.cli.export_model -h

List detectors

python -c "import detect; print(detect.available_detectors())"

List models (registry + installed)

python -m detect.cli.detect_video --list-models
python -m detect.cli.export_model --list-models


⸻

Detection CLI (pip)

In-memory only (default; saves nothing):

python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n

Save JSON + frames + annotated video (example):

python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_seg \
  --weights yolo26n-seg \
  --json --frames \
  --save-video annotated.mp4 \
  --out-dir out --run-name run_seg


⸻

Export CLI (pip)

Requires export extras:

pip install "detect[export]"

Export ONNX:

python -m detect.cli.export_model \
  --weights yolo26n \
  --formats onnx \
  --out-dir models/exports --run-name y26_onnx

Validate ONNX:

python -c "import onnx; m=onnx.load('models/exports/y26_onnx/yolo26n.onnx'); onnx.checker.check_model(m); print('ONNX OK')"

Confirm ONNXRuntime loads:

python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"


⸻

Python usage (import)

After installing with pip install detect, you can use it in code.

Quick sanity check

python -c "import detect; print(detect.available_detectors())"

Run detection from a Python file

Create run_detect.py:

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

payload = res.payload
print(payload["schema_version"], len(payload["frames"]))
print(res.paths)  # populated only if you enable saving artifacts

Run:

python run_detect.py


⸻

Install from GitHub (uv)

Use this if you are developing locally or want reproducible project environments.

Repo:
	•	https://github.com/Surya-Rayala/VideoPipeline-detection.git

Install uv:
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

uv --version

Clone + install deps:

git clone https://github.com/Surya-Rayala/VideoPipeline-detection.git
cd VideoPipeline-detection
uv sync

Optional installs (uv extras)

uv sync --extra export
uv sync --extra tf
uv sync --extra openvino
uv sync --extra coreml


⸻

CLI usage (uv)

When running from the repo with uv, use uv run:

Global help

uv run python -m detect.cli.detect_video -h
uv run python -m detect.cli.export_model -h

List detectors

uv run python -c "import detect; print(detect.available_detectors())"

List models (registry + installed)

uv run python -m detect.cli.detect_video --list-models
uv run python -m detect.cli.export_model --list-models


⸻

Detection CLI (uv)

In-memory only (default; saves nothing):

uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n

Save JSON only:

uv run python -m detect.cli.detect_video \
  --video <in.mp4> \
  --detector yolo_bbox \
  --weights yolo26n \
  --json \
  --out-dir out --run-name run_json


⸻

Export CLI (uv)

Requires:

uv sync --extra export

Export ONNX:

uv run python -m detect.cli.export_model \
  --weights yolo26n \
  --formats onnx \
  --out-dir models/exports --run-name y26_onnx

Validate ONNX:

uv run python -c "import onnx; m=onnx.load('models/exports/y26_onnx/yolo26n.onnx'); onnx.checker.check_model(m); print('ONNX OK')"

Confirm ONNXRuntime loads:

uv run python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"


⸻

TensorRT / engine export and run notes (important)

Exporting engine (TensorRT) typically requires an NVIDIA GPU + CUDA + TensorRT installed and version-compatible.

Export TensorRT engine:

uv run python -m detect.cli.export_model \
  --weights yolo26n \
  --formats engine \
  --device 0 \
  --out-dir models/exports --run-name y26_trt

Run / sanity-check the exported engine (Ultralytics predict):

uv run python -c "from ultralytics import YOLO; m=YOLO('models/exports/y26_trt/yolo26n.engine'); r=m.predict(source='in.mp4', device=0, verbose=False); print('OK', len(r))"


⸻

CLI argument reference

detect.cli.detect_video

Required
	•	--video <path>: Path to the input video file.
	•	--detector <name>: Detector type (yolo_bbox, yolo_pose, or yolo_seg).
	•	--weights <id|path|url>: Registry key, local weights path, or URL to weights.

Detection options
	•	--classes <ids>: Filter to specific class IDs (comma/semicolon-separated), or omit for all classes.
	•	--conf-thresh <float>: Confidence threshold for detections (default 0.25).
	•	--imgsz <int>: Inference image size used by the backend (default 640).
	•	--device <str>: Device selector (e.g., auto, cpu, mps, 0).
	•	--half: Enable FP16 inference where supported.

Artifact saving (opt-in)
	•	--json: Save detections.json under the run directory.
	•	--frames: Save extracted frames as images under the run directory.
	•	--save-video <name.mp4>: Save an annotated video under the run directory.
	•	--display: Show live visualization window while running (press q to quit).

Output control
	•	--out-dir <dir>: Output root directory used only if saving artifacts (default out).
	•	--run-name <name>: Run folder name inside out-dir (auto-derived if omitted).

Model registry / downloads
	•	--models-dir <dir>: Directory where models are stored/downloaded (default models).
	•	--no-download: Disable automatic download for registry keys/URLs.

Misc
	•	--no-progress: Disable progress bar output.
	•	--list-models: Print registry + installed models then exit.

⸻

detect.cli.export_model

Required
	•	--weights <id|path|url>: Registry key, local weights path, or URL to weights.

Export options
	•	--formats <list>: Comma/semicolon-separated export formats (default onnx).
	•	--imgsz <int|H,W>: Export image size as an int or H,W pair (default 640).
	•	--device <str>: Export device selector (e.g., cpu, mps, 0).
	•	--half: Enable FP16 export where supported.
	•	--int8: Enable INT8 quantization (format/toolchain-dependent).
	•	--data <yaml>: Dataset YAML for INT8 calibration (when required).
	•	--fraction <float>: Fraction of dataset used for calibration (default 1.0).
	•	--dynamic: Enable dynamic shapes where supported.
	•	--batch <int>: Export batch size (default 1).
	•	--opset <int>: ONNX opset version (ONNX only).
	•	--simplify: Simplify the ONNX graph (ONNX only).
	•	--workspace <int>: TensorRT workspace size in GB (TensorRT only).
	•	--nms: Add NMS to exported model when supported by format/backend.

Output control
	•	--out-dir <dir>: Output root directory for exports (default models/exports).
	•	--run-name <name>: Export run folder name inside out-dir.

Model registry / downloads
	•	--models-dir <dir>: Directory where models are stored/downloaded (default models).
	•	--no-download: Disable automatic download for registry keys/URLs.

Misc
	•	--list-models: Print registry + installed models then exit.

⸻

License

MIT License. See LICENSE.