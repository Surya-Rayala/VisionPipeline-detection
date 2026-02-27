from __future__ import annotations

"""
detect.registry.registry
------------------------

Model registry + weights resolver.

Accepts a "weights" identifier that can be:
  • A local path -> returned as-is (must exist)
  • A URL        -> downloaded to models_dir
  • A registry key (e.g., "yolo11x", "yolo11n-seg", "yolo26n") -> looked up and downloaded if missing

Registry sources are merged in this order (later overrides earlier):
  1) DEFAULT_MODEL_REGISTRY (in-code defaults)
  2) detect/registry/default.json            (package default; optional but recommended)
  3) ~/.detect/registry.json                 (user-local override; optional)
  4) file pointed to by DETECT_MODEL_REGISTRY env var (optional)

Registry entry schema (extensible, backward compatible):
  "<key>": {
    "url": "https://.../weights.pt",
    "filename": "weights.pt",                  # optional; derived from URL if omitted
    "detector": "yolo_bbox|yolo_pose|yolo_seg" # optional; advisory
    "backend": "ultralytics"                   # optional; advisory
    "sha256": "<hex>"                          # optional; verify after download
  }
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Union
from urllib.parse import urlparse
import urllib.request


# Default directory to place downloaded/managed models
DEFAULT_MODELS_DIR = Path(os.getenv("DETECT_MODELS_DIR", "models")).resolve()

# In-code defaults (normally empty; keep for programmatic injection)
DEFAULT_MODEL_REGISTRY: Dict[str, Dict] = {}

# File suffixes we treat as "model artifacts" for local discovery and/or download.
_MODEL_SUFFIXES: Set[str] = {".pt", ".onnx", ".engine", ".bin", ".tflite", ".mlpackage"}

# SAM3 weights are not auto-downloaded by Ultralytics; users must manually download.
# We detect these stems to provide a clear error message.
_SAM3_STEMS: Set[str] = {"sam3"}


# --- Internal helpers ---
def _read_json(p: Path) -> Dict:
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_registry_paths() -> Iterable[Path]:
    """Yield candidate JSON registry files (package default, user-level, env override)."""
    # Package default registry lives next to this file
    here = Path(__file__).resolve().parent
    yield here / "default.json"

    # User-level override
    yield Path.home() / ".detect" / "registry.json"

    # Env override
    env = os.getenv("DETECT_MODEL_REGISTRY")
    if env:
        yield Path(env)


def load_registry() -> Dict[str, Dict]:
    """Return merged registry dict from defaults + JSON files."""
    reg = dict(DEFAULT_MODEL_REGISTRY)
    for p in _load_registry_paths():
        reg.update(_read_json(p))
    return reg


def list_registered_models(detector: Optional[str] = None, backend: Optional[str] = None) -> Dict[str, Dict]:
    """List registered model keys (optionally filtered by detector/backend)."""
    reg = load_registry()

    def ok(v: Dict) -> bool:
        if detector and v.get("detector") not in (None, detector):
            return False
        if backend and v.get("backend") not in (None, backend):
            return False
        return True

    return {k: v for k, v in reg.items() if ok(v)}


def list_installed_models(models_dir: Union[str, Path] = DEFAULT_MODELS_DIR) -> Dict[str, str]:
    """List local model files in models_dir (by filename)."""
    d = Path(models_dir)
    out: Dict[str, str] = {}
    if not d.exists():
        return out
    for p in d.glob("*"):
        if p.is_file() and p.suffix.lower() in _MODEL_SUFFIXES:
            out[p.name] = str(p.resolve())
    return out

def _attempt_ultralytics_asset_download(name: str, dst: Path, *, allow_download: bool) -> Optional[Path]:
    """Best-effort download via Ultralytics' own GitHub assets helper.

    This supports names like 'rtdetr-l.pt', 'yolov10n.pt', etc.

    Returns the resolved path if downloaded/found, else None.
    """
    if not allow_download:
        return None

    # SAM3 is not distributed via ultralytics/assets auto-download.
    if Path(name).stem.lower().startswith("sam3"):
        return None

    try:
        # Ultralytics provides a downloader that knows how to map known pretrained filenames
        # (e.g. 'rtdetr-l.pt', 'yolov10n.pt', etc.) to the correct GitHub assets URL.
        # Different Ultralytics versions expose different kwargs, so we avoid passing
        # repo/release and just let it use its defaults.
        from ultralytics.utils.downloads import attempt_download_asset  # type: ignore

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Ask Ultralytics to download the asset by name.
        out = attempt_download_asset(name)
        p = Path(out)
        if not p.exists():
            return None

        # Move/copy into our models_dir location if needed.
        try:
            if p.resolve() != dst.resolve():
                dst.write_bytes(p.read_bytes())
                # Best-effort cleanup of the original download location (often CWD).
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            # As a fallback, try rename (may fail across filesystems)
            try:
                p.replace(dst)
            except Exception:
                return p.resolve()

        return dst.resolve() if dst.exists() else p.resolve()
    except Exception:
        return None


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return bool(u.scheme and u.netloc)
    except Exception:
        return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> Path:
    """Download to a temp file then atomically move into place."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent)) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as r, tmp_path.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp_path.replace(dst)
        return dst.resolve()
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def resolve_weights_path(
    identifier: Union[str, Path],
    *,
    models_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
    detector: Optional[str] = None,
    backend: Optional[str] = None,
    allow_download: bool = True,
) -> Path:
    """
    Resolve a path/URL/registry key into a guaranteed local file path.

    Behavior:
      • If identifier is a path-like -> it must exist.
      • If identifier is a URL     -> download to models_dir (if missing).
      • Else, treat as registry key -> lookup + download to models_dir (if missing).

    Raises FileNotFoundError / KeyError / RuntimeError on problems.
    """
    models_dir = Path(models_dir)

    # 1) Path-like? (contains separators or is already a Path)
    if isinstance(identifier, Path) or any(sep in str(identifier) for sep in (os.sep, "/", "\\")):
        p = Path(identifier)
        if not p.exists():
            raise FileNotFoundError(f"Weights path not found: {p}")
        return p.resolve()

    s = str(identifier).strip()

    # 2) URL?
    if _is_url(s):
        filename = Path(urlparse(s).path).name or "weights.pt"
        dst = models_dir / filename
        if dst.exists():
            return dst.resolve()
        if not allow_download:
            raise FileNotFoundError(f"Download disabled, and file not found: {dst}")
        return _download(s, dst)

    # 3) Registry key
    reg = load_registry()
    if s not in reg:
        # Allow bare filename present under models_dir (e.g., "yolo11x.pt")
        maybe = models_dir / s
        if maybe.exists():
            return maybe.resolve()
        # SAM3 weights are not auto-downloaded; require a local path.
        if backend == "ultralytics" and Path(s).stem.lower().startswith("sam3"):
            raise FileNotFoundError(
                "SAM3 weights are not auto-downloaded. Please download the SAM3 .pt file manually "
                "(Ultralytics SAM-3 docs) and pass a local path, e.g. --weights /path/to/sam3.pt."
            )
        # If this looks like a bare Ultralytics pretrained filename (e.g. 'rtdetr-l.pt'),
        # try to download it via Ultralytics' own asset downloader when backend is ultralytics.
        suf = Path(s).suffix.lower()
        if backend == "ultralytics" and suf in _MODEL_SUFFIXES:
            dst = models_dir / Path(s).name
            got = _attempt_ultralytics_asset_download(Path(s).name, dst, allow_download=allow_download)
            if got is not None:
                return got
        raise KeyError(
            f"Unknown model key '{s}'. Add it to a registry JSON, pass a valid path/URL, or (Ultralytics) pass a known pretrained filename like 'rtdetr-l.pt'."
        )

    entry = reg[s]
    url = entry.get("url")
    if not url:
        raise KeyError(f"Registry entry '{s}' missing 'url'.")

    filename = entry.get("filename") or Path(urlparse(url).path).name or f"{s}.pt"
    dst = models_dir / filename

    if backend == "ultralytics" and Path(filename).stem.lower().startswith("sam3"):
        # Even if present in a registry, SAM3 typically requires manual download/auth.
        # Enforce local path to avoid confusing failures.
        if dst.exists():
            return dst.resolve()
        raise FileNotFoundError(
            "SAM3 weights are not auto-downloaded. Please place the SAM3 .pt file in your models_dir "
            f"({models_dir}) or pass a local path via --weights."
        )

    # Optional compatibility hints (warn only)
    if detector and entry.get("detector") and entry["detector"] != detector:
        print(
            f"[warn] Registry entry '{s}' is marked for detector '{entry['detector']}', "
            f"but requested '{detector}'. Proceeding."
        )
    if backend and entry.get("backend") and entry["backend"] != backend:
        print(
            f"[warn] Registry entry '{s}' is marked for backend '{entry['backend']}', "
            f"but requested '{backend}'. Proceeding."
        )

    if dst.exists():
        want = (entry.get("sha256") or "").strip()
        if want:
            have = _sha256_file(dst)
            if want.lower() != have.lower():
                print("[warn] SHA256 mismatch on existing file; re-downloading.")
                try:
                    dst.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                return dst.resolve()
        else:
            return dst.resolve()

    if not allow_download:
        raise FileNotFoundError(f"Model '{s}' not present at {dst} and download disabled.")

    got = _download(url, dst)
    want = (entry.get("sha256") or "").strip()
    if want:
        have = _sha256_file(got)
        if want.lower() != have.lower():
            try:
                got.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"Checksum failed for '{s}' after download.")
    return got.resolve()


__all__ = [
    "DEFAULT_MODELS_DIR",
    "DEFAULT_MODEL_REGISTRY",
    "load_registry",
    "list_registered_models",
    "list_installed_models",
    "resolve_weights_path",
]