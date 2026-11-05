"""Shared ONNX Runtime helpers with GPU (CUDA) and CPU fallback providers."""
from __future__ import annotations

import onnxruntime as ort

from .config import setup_cuda_dlls

setup_cuda_dlls()


def cuda_providers() -> list[str]:
    """Return execution providers, prioritizing CUDA GPU with automatic CPU fallback."""
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = available or ["CPUExecutionProvider"]
    return providers


get_inference_providers = cuda_providers


def verify_cuda_session(session: ort.InferenceSession, name: str) -> None:
    """Check active execution providers for the ONNX session (GPU preferred, CPU fallback permitted)."""
    active = session.get_providers()
    if active and "CUDAExecutionProvider" in active:
        # Running on CUDA GPU
        return
    # CPU Fallback
    print(f"[Face Engine] Info: {name} running on CPU (active providers: {active})")
