"""Project paths and constants."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENROLLED_DIR = ROOT / "data" / "enrolled"     # multi-angle manual enrollment images
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
ALIGNED_DIR = OUTPUT_DIR / "aligned"
YOLO_WEIGHTS = MODELS_DIR / "yolov11n-face.pt"
SCRFD_WEIGHTS = MODELS_DIR / "scrfd_10g_bnkps.onnx"
MINIFASNET_WEIGHTS = MODELS_DIR / "2.7_80x80_MiniFASNetV2.pth"
GALLERY_PATH = OUTPUT_DIR / "gallery.npz"


def get_arcface_weights() -> Path:
    """Find ArcFace weights in buffalo_l/ subdir or models root."""
    candidates = [
        MODELS_DIR / "buffalo_l" / "w600k_r50.onnx",
        MODELS_DIR / "w600k_r50.onnx",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


ARCFACE_WEIGHTS = get_arcface_weights()

ARCFACE_INPUT = 112
SCRFD_DET_SIZE = 640   # 640x640 optimal balance: 12.9ms latency (16x faster than 1280) with high landmark precision
YOLO_CONF = 0.25       # Calibrated confidence threshold for face detection
YOLO_IMGSZ = 1280
SAME_PERSON_THRESHOLD = 0.70   # Strict cosine similarity matching threshold
SPOOF_THRESHOLD = 0.80         # Balanced MiniFASNetV2 liveness score cutoff (real faces score >0.85; blocks screen attacks)
STABILITY_THRESHOLD = 0.32
REQUIRED_FRAMES = 4
MIN_FACE_SIZE = 16     # Minimum physical face dimension (px) to reject background noise specks

# Context Auto-Scaling for Anti-Spoofing:
# Distant faces (<= 60px) use tighter 2.2x context to preserve face resolution in 80x80 crop.
# Close-up faces (>= 280px) use wider 3.5x context to capture neck, holding hands, screen borders.
SPOOF_DYNAMIC_SCALE = True
SPOOF_MIN_SCALE = 2.2
SPOOF_MAX_SCALE = 3.5
SPOOF_MIN_FACE_H = 60.0
SPOOF_MAX_FACE_H = 280.0


def setup_cuda_dlls():
    """Ensure ONNX Runtime can find CUDA/cuDNN DLLs bundled inside torch/lib on Windows."""
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
    except Exception:
        pass


def load_env():
    """Load KEY=VALUE pairs from the project .env (if present) into os.environ."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# Keep console output (including third-party loggers) from crashing on
# non-ASCII characters when running on Windows code pages.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

setup_cuda_dlls()
load_env()
