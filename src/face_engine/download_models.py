"""Automated model downloader for cloud deployments (Streamlit Community Cloud, Hugging Face, etc.)."""
from __future__ import annotations

import sys
import zipfile
import urllib.request
from pathlib import Path

from . import config

BUFFALO_L_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def ensure_models_exist(progress_callback=None) -> None:
    """Ensure all required model weights exist locally.
    
    If w600k_r50.onnx (ArcFace) is missing (due to GitHub 100MB file limit),
    it downloads it automatically from official releases.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    buffalo_dir = config.MODELS_DIR / "buffalo_l"
    buffalo_dir.mkdir(parents=True, exist_ok=True)

    arcface_weight = config.ARCFACE_WEIGHTS
    if not arcface_weight.exists():
        msg = "Downloading ArcFace weights (w600k_r50.onnx)... This happens only once."
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"[Face Engine] {msg}")

        zip_path = config.MODELS_DIR / "buffalo_l.zip"
        req = urllib.request.Request(BUFFALO_L_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

        # Extract w600k_r50.onnx to buffalo_l/ and ensure destination is valid
        import shutil
        with zipfile.ZipFile(zip_path, "r") as z:
            for item in z.namelist():
                if item.endswith("w600k_r50.onnx"):
                    z.extract(item, buffalo_dir)
                    extracted = buffalo_dir / item
                    # Ensure it is at buffalo_dir / w600k_r50.onnx
                    canonical = buffalo_dir / "w600k_r50.onnx"
                    if extracted.exists() and extracted != canonical:
                        shutil.move(str(extracted), str(canonical))

        # Clean up temporary zip
        if zip_path.exists():
            zip_path.unlink()

        # Update config.ARCFACE_WEIGHTS
        config.ARCFACE_WEIGHTS = config.get_arcface_weights()
        print(f"[Face Engine] ArcFace weights ready at {config.ARCFACE_WEIGHTS}")

