# Face Engine

A modern, cloud-deployable biometric face recognition and liveness verification platform built with Streamlit, ArcFace, SCRFD, and MiniFASNetV2.

**Pipeline: YOLO11-Face Detection → SCRFD 5-Point Alignment → ArcFace Embedding → MiniFASNetV2 Anti-Spoofing → Multi-Angle Gallery Identification.**

---

## Architecture & Features

- **Unified Web Interface with Mode Toggler**: Single unified camera workflow with an instant toggle switch (`st.toggle("➕ Enroll Mode")`) to switch between live inference prediction and identity enrollment right on the same feed.
- **Hardware Acceleration**: Prioritizes CUDA GPU (NVIDIA GeForce GTX 1660 SUPER) with seamless, automatic CPU fallback for standard cloud deployment containers.
- **Presentation Attack Detection (Anti-Spoofing)**: Pure PyTorch MiniFASNetV2 with proximity-aware dynamic context auto-scaling ($2.2\times$ to $3.5\times$, cutoff `0.80`), rejecting digital screens, tablets, and printed photo replays.
- **3D Head Pose & Liveness Verification**: Real-time estimation of yaw, pitch, and roll with verified liveness checks before saving templates.
- **Gallery Management**: Manage identities in `outputs/gallery.npz`, preview thumbnail photo archives, rename, and delete profiles on the fly.

---

## Directory Layout

```
.
├── data/
│   └── enrolled/                # Multi-angle 16-frame enrollment profiles per identity
│       ├── Vihaan/
│       ├── Ashutosh/
│       └── Abishek/
├── models/                      # Deep Learning Model Weights
│   ├── yolov11n-face.pt         # Face bounding box detection
│   ├── scrfd_10g_bnkps.onnx     # 5-point facial keypoint localization
│   ├── buffalo_l/w600k_r50.onnx # ArcFace 512-d facial embedding
│   └── 2.7_80x80_MiniFASNetV2.pth # Pure PyTorch anti-spoof / liveness checker
├── src/face_engine/             # Core Python package
│   ├── align.py                 # SCRFD keypoint detection & ArcFace affine alignment
│   ├── antispoof.py             # MiniFASNetV2 GPU engine with dynamic auto-scaling
│   ├── config.py                # Hyperparameters, paths & CUDA environment
│   ├── detect.py                # YOLO face detector utilities
│   ├── embed.py                 # ArcFace feature extraction
│   ├── gallery.py               # Template store & multi-profile cosine matching
│   ├── onnx_utils.py            # ONNX Runtime provider management & CPU fallback
│   └── pose.py                  # 3D head pose estimation (yaw, pitch, roll)
├── outputs/                     # Generated galleries & logs
│   ├── gallery.npz              # Master facial embedding templates
│   └── gallery.json             # Identity metadata
├── app.py                       # Main Streamlit web application
├── run.bat                      # Windows one-click launcher
├── requirements.txt
└── pyproject.toml
```

---

## Quickstart

### 1. Requirements & Setup
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
# Or on Windows, double-click:
run.bat
```
Open **`http://localhost:8501`** in any web browser.

---

## Deployment (Streamlit Cloud, Docker, AWS)

Because the application uses native Streamlit browser camera input (`st.camera_input`), it runs out of the box in cloud environments without requiring a physical monitor or direct server camera access:

1. **Streamlit Community Cloud**:
   - Main file path: `app.py`
   - Requirements file: `requirements.txt`
2. **Docker / Cloud Container**:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

---

## Testing

```bash
pytest
```
- `tests/test_smoke.py`: Core pipeline sanity & gallery template checks.
- `tests/test_dynamic_scale.py`: Anti-spoofing dynamic scale bounds, border clamping, and GPU/CPU execution checks.
