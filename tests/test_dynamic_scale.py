"""Unit tests for Resolution-Aware Dynamic Context Auto-Scaling and MiniFASNetV2."""
import numpy as np
import pytest
import torch
import cv2

from face_engine import config
from face_engine.antispoof import (
    MiniFASNetV2Checker,
    SpoofResult,
    compute_dynamic_scale,
    crop_face_minivision,
)


def test_dynamic_scale_bounds():
    """Verify compute_dynamic_scale clamps correctly at far and close ranges."""
    # Far/small faces (h <= 60px) should return min_scale (2.2)
    assert compute_dynamic_scale(20.0) == 2.2
    assert compute_dynamic_scale(60.0) == 2.2

    # Close-up faces (h >= 280px) should return max_scale (3.5)
    assert compute_dynamic_scale(280.0) == 3.5
    assert compute_dynamic_scale(450.0) == 3.5

    # Intermediate sizes should interpolate smoothly
    s_100 = compute_dynamic_scale(100.0)
    s_145 = compute_dynamic_scale(145.0)
    s_200 = compute_dynamic_scale(200.0)

    assert 2.2 < s_100 < s_145 < s_200 < 3.5
    # At h=145 (typical webcam distance), scale should be approx 2.70 (standard Minivision)
    assert abs(s_145 - 2.70) < 0.05


def test_crop_face_minivision_dimensions():
    """Test crop_face_minivision produces exact (80, 80, 3) across various distances."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Distant face (40x50)
    crop_far = crop_face_minivision(img, (200, 200, 240, 250), dynamic_scale=True)
    assert crop_far.shape == (80, 80, 3)

    # Medium face (140x160)
    crop_mid = crop_face_minivision(img, (500, 200, 640, 360), dynamic_scale=True)
    assert crop_mid.shape == (80, 80, 3)

    # Huge close-up face (350x400)
    crop_close = crop_face_minivision(img, (400, 100, 750, 500), dynamic_scale=True)
    assert crop_close.shape == (80, 80, 3)


def test_crop_face_minivision_boundary_clamping():
    """Verify face crops near image edges never crash or go out of bounds."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Top-left corner
    c_tl = crop_face_minivision(img, (2, 2, 80, 100), dynamic_scale=True)
    assert c_tl.shape == (80, 80, 3)

    # Bottom-right corner
    c_br = crop_face_minivision(img, (560, 400, 638, 478), dynamic_scale=True)
    assert c_br.shape == (80, 80, 3)

    # Face spanning almost entire frame height
    c_full = crop_face_minivision(img, (50, 10, 500, 470), dynamic_scale=True)
    assert c_full.shape == (80, 80, 3)


def test_minifasnetv2_checker_cuda_real_face():
    """Test MiniFASNetV2Checker runs on CUDA and evaluates real enrolled faces."""
    assert torch.cuda.is_available(), "CUDA must be available for testing"
    checker = MiniFASNetV2Checker(weights_path=config.MINIFASNET_WEIGHTS, threshold=config.SPOOF_THRESHOLD)

    # Check device is CUDA
    assert checker.device.type == "cuda"
    assert next(checker.model.parameters()).is_cuda

    img_path = config.ROOT / "data" / "enrolled" / "Ashutosh" / "frame_1_front.jpg"
    img = cv2.imread(str(img_path))
    assert img is not None
    h, w = img.shape[:2]

    # Test single check with dynamic scale
    res = checker.check(img, (0, 0, w, h))
    assert isinstance(res, SpoofResult)
    assert isinstance(res.is_real, bool)
    assert 0.0 <= res.score <= 1.0

    # Test batched check
    batch_res = checker.check_batch(img, [(0, 0, w, h), (10, 10, w - 10, h - 10)])
    assert len(batch_res) == 2
    for r in batch_res:
        assert isinstance(r, SpoofResult)
        assert 0.0 <= r.score <= 1.0

