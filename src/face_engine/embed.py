"""ArcFace embedding helpers (strictly GPU)."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from pathlib import Path
from .onnx_utils import cuda_providers, verify_cuda_session


class FaceEmbedder:
    def __init__(self, weights_path: str | Path):
        weights = Path(weights_path)
        if not weights.exists():
            from .config import get_arcface_weights
            weights = get_arcface_weights()
            if not weights.exists():
                from .download_models import ensure_models_exist
                ensure_models_exist()
                weights = get_arcface_weights()

        if not weights.exists():
            raise FileNotFoundError(f"ArcFace model weights file not found: {weights_path}")

        self.sess = ort.InferenceSession(str(weights), providers=cuda_providers())
        verify_cuda_session(self.sess, "ArcFace")
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a unit-norm 512-D embedding for a single 112x112 BGR face crop."""
        return self.embed_batch([image_bgr])[0]

    def embed_batch(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Return unit-norm (N, 512) embeddings for multiple 112x112 BGR face crops in a single parallel GPU forward pass."""
        if not images_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        blob = cv2.dnn.blobFromImages(
            images_bgr,
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        vecs = self.sess.run(None, {self.input_name: blob})[0]  # shape (N, 512)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return (vecs / norms).astype(np.float32)
