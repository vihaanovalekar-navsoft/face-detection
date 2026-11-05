"""Face anti-spoofing (liveness / presentation-attack detection) using MiniFASNetV2.

Based on Minivision's Silent-Face-Anti-Spoofing:
  Architecture: MiniFASNetV2 with Fourier auxiliary supervision
  Input: 80x80 face crop with proximity-aware dynamic context auto-scaling (2.2x to 3.5x)
  Execution: Strictly CUDA GPU (GeForce GTX 1660 SUPER, zero CPU fallback)
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import (
    MINIFASNET_WEIGHTS,
    SPOOF_DYNAMIC_SCALE,
    SPOOF_MAX_FACE_H,
    SPOOF_MAX_SCALE,
    SPOOF_MIN_FACE_H,
    SPOOF_MIN_SCALE,
    SPOOF_THRESHOLD,
    setup_cuda_dlls,
)

setup_cuda_dlls()


@dataclass(frozen=True)
class SpoofResult:
    """Outcome of an anti-spoofing / liveness check."""

    is_real: bool
    score: float
    method: str = "minifasnetv2"


# ==============================================================================
# MiniFASNetV2 Architecture Definition (PyTorch GPU)
# ==============================================================================

class Conv_block(nn.Module):
    """Convolution -> BatchNorm -> PReLU building block."""

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class Linear_block(nn.Module):
    """Convolution -> BatchNorm block without activation."""

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class Depth_Wise(nn.Module):
    """Depthwise separable block with optional residual connection."""

    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        short_cut = x if self.residual else None
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return (short_cut + x) if self.residual else x


class Residual(nn.Module):
    """Sequence of residual depthwise separable blocks."""

    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = [
            Depth_Wise(
                c1[i], c2[i], c3[i], residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups
            )
            for i in range(num_block)
        ]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Channel configuration for MiniFASNetV2 ('1.8M_')
_KEEP_DICT_V2 = [
    32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
    52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512
]


class MiniFASNet(nn.Module):
    """MiniFASNet backbone network."""

    def __init__(
        self,
        keep: list[int],
        embedding_size: int = 128,
        conv6_kernel: tuple[int, int] = (5, 5),
        drop_p: float = 0.2,
        num_classes: int = 3,
        img_channel: int = 3,
    ):
        super().__init__()
        self.embedding_size = embedding_size

        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        return self.prob(out)


def build_minifasnet_v2(
    embedding_size: int = 128,
    conv6_kernel: tuple[int, int] = (5, 5),
    drop_p: float = 0.2,
    num_classes: int = 3,
) -> MiniFASNet:
    """Build MiniFASNetV2 architecture with 80x80 kernel parameters."""
    return MiniFASNet(
        _KEEP_DICT_V2,
        embedding_size=embedding_size,
        conv6_kernel=conv6_kernel,
        drop_p=drop_p,
        num_classes=num_classes,
    )


# ==============================================================================
# Bounding Box Expansion & Resolution-Aware Dynamic Context Cropping
# ==============================================================================

def compute_dynamic_scale(
    box_h: float,
    min_scale: float = SPOOF_MIN_SCALE,
    max_scale: float = SPOOF_MAX_SCALE,
    min_face_h: float = SPOOF_MIN_FACE_H,
    max_face_h: float = SPOOF_MAX_FACE_H,
) -> float:
    """Calculate adaptive context expansion scale based on face height / camera proximity.

    - Distant/small faces (h <= min_face_h): returns min_scale (2.2x), keeping the face
      prominent inside the 80x80 patch so facial texture is not downscaled into blur.
    - Close-up faces (h >= max_face_h): returns max_scale (3.5x), capturing wide surrounding
      context (neck, chest, ears, hair, and smartphone screen bezels/borders).
    - Intermediate sizes: smoothly interpolates linearly between min_scale and max_scale.
    """
    if max_face_h <= min_face_h:
        return float(min_scale)
    norm_h = max(0.0, min(1.0, (float(box_h) - float(min_face_h)) / (float(max_face_h) - float(min_face_h))))
    return float(min_scale + norm_h * (max_scale - min_scale))


def _to_xyxy(face_box, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Normalize a face box to (x1, y1, x2, y2) clipped to frame bounds."""
    fh, fw = shape[:2]
    box = [float(v) for v in face_box]
    if len(box) == 4 and box[2] > box[0] and box[3] > box[1] and box[2] <= fw and box[3] <= fh:
        x1, y1, x2, y2 = box
    elif len(box) == 4:
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        raise ValueError(f"Unsupported face_box: {face_box}")

    x1 = int(max(0, min(fw - 1, x1)))
    y1 = int(max(0, min(fh - 1, y1)))
    x2 = int(max(x1 + 1, min(fw, x2)))
    y2 = int(max(y1 + 1, min(fh, y2)))
    return x1, y1, x2, y2


def crop_face_minivision(
    image: np.ndarray,
    face_box,
    scale: float | None = None,
    out_size: tuple[int, int] = (80, 80),
    dynamic_scale: bool = SPOOF_DYNAMIC_SCALE,
    min_scale: float = SPOOF_MIN_SCALE,
    max_scale: float = SPOOF_MAX_SCALE,
) -> np.ndarray:
    """Crop face box with Minivision scale expansion and resize to (80, 80).

    Args:
        image: Source image frame (H, W, C).
        face_box: Bounding box as (x1, y1, x2, y2) or (x, y, w, h).
        scale: Fixed expansion scale. If specified, overrides dynamic auto-scaling.
        out_size: Target output dimensions (width, height), default (80, 80).
        dynamic_scale: If True and scale is None, automatically scales context from
            min_scale (2.2x for far faces) to max_scale (3.5x for close faces)
            based on face height / proximity.
        min_scale: Scale for small / far faces (default 2.2).
        max_scale: Scale for close-up faces (default 3.5).
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = _to_xyxy(face_box, (h_img, w_img))
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    if scale is not None:
        effective_scale = float(scale)
    elif dynamic_scale:
        effective_scale = compute_dynamic_scale(
            box_h,
            min_scale=min_scale,
            max_scale=max_scale,
            min_face_h=SPOOF_MIN_FACE_H,
            max_face_h=SPOOF_MAX_FACE_H,
        )
    else:
        effective_scale = 2.7

    # Clamp effective scale so expanded crop cannot exceed frame dimensions
    effective_scale = min((h_img - 1) / box_h, min((w_img - 1) / box_w, effective_scale))
    new_width = box_w * effective_scale
    new_height = box_h * effective_scale
    center_x = box_w / 2.0 + x1
    center_y = box_h / 2.0 + y1

    left_top_x = center_x - new_width / 2.0
    left_top_y = center_y - new_height / 2.0
    right_bottom_x = center_x + new_width / 2.0
    right_bottom_y = center_y + new_height / 2.0

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > w_img - 1:
        left_top_x -= right_bottom_x - w_img + 1
        right_bottom_x = w_img - 1
    if right_bottom_y > h_img - 1:
        left_top_y -= right_bottom_y - h_img + 1
        right_bottom_y = h_img - 1

    x1_c = max(0, min(w_img - 1, int(left_top_x)))
    y1_c = max(0, min(h_img - 1, int(left_top_y)))
    x2_c = min(w_img, max(x1_c + 1, int(right_bottom_x) + 1))
    y2_c = min(h_img, max(y1_c + 1, int(right_bottom_y) + 1))

    crop = image[y1_c:y2_c, x1_c:x2_c]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return cv2.resize(image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)], out_size)
    return cv2.resize(crop, out_size)


# ==============================================================================
# MiniFASNetV2 Anti-Spoof Checker (GPU Engine)
# ==============================================================================

class MiniFASNetV2Checker:
    """Face anti-spoofing checker backed by Minivision's MiniFASNetV2 on CUDA."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        threshold: float = SPOOF_THRESHOLD,
        device: str | None = None,
        dynamic_scale: bool = SPOOF_DYNAMIC_SCALE,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            print("[Face Engine] Info: CUDA not available. MiniFASNetV2 using CPU fallback.")
            device = "cpu"

        if weights_path is None:
            weights_path = MINIFASNET_WEIGHTS
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"MiniFASNetV2 weights not found at {self.weights_path}. "
                f"Ensure 2.7_80x80_MiniFASNetV2.pth exists in models/."
            )

        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.dynamic_scale = bool(dynamic_scale)

        # Initialize network architecture
        self.model = build_minifasnet_v2(embedding_size=128, conv6_kernel=(5, 5), num_classes=3)

        # Load weights checkpoint
        state_dict = torch.load(str(self.weights_path), map_location=self.device)
        new_sd = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v)
            for k, v in state_dict.items()
        )
        self.model.load_state_dict(new_sd)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def check(self, frame: np.ndarray, face_box, scale: float | None = None) -> SpoofResult:
        """Check whether a single face bounding box in ``frame`` is real.

        Returns SpoofResult(is_real=True/False, score=real_face_probability).
        """
        if frame is None or face_box is None:
            return SpoofResult(is_real=False, score=0.0, method="invalid-input")

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = _to_xyxy(face_box, (fh, fw))
        w = x2 - x1
        h = y2 - y1
        if w < 8 or h < 8:
            return SpoofResult(is_real=False, score=0.0, method="too-small")

        try:
            patch = crop_face_minivision(
                frame,
                (x1, y1, x2, y2),
                scale=scale,
                out_size=(80, 80),
                dynamic_scale=self.dynamic_scale if scale is None else False,
            )
            # Minivision normalization: BGR transpose to CHW float32 [0..255]
            tensor = torch.from_numpy(patch.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            real_score = float(probs[1])

            is_real = real_score >= self.threshold
            return SpoofResult(is_real=bool(is_real), score=real_score, method="minifasnetv2")
        except Exception:
            return SpoofResult(is_real=False, score=0.0, method="error")

    @torch.inference_mode()
    def check_batch(self, frame: np.ndarray, face_boxes: list, scale: float | None = None) -> list[SpoofResult]:
        """Check multiple face bounding boxes in a single batched GPU forward pass."""
        if frame is None or not face_boxes:
            return []

        fh, fw = frame.shape[:2]
        valid_indices = []
        tensors = []
        results = [SpoofResult(is_real=False, score=0.0, method="invalid-box")] * len(face_boxes)

        for i, box in enumerate(face_boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = _to_xyxy(box, (fh, fw))
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                results[i] = SpoofResult(is_real=False, score=0.0, method="too-small")
                continue
            try:
                patch = crop_face_minivision(
                    frame,
                    (x1, y1, x2, y2),
                    scale=scale,
                    out_size=(80, 80),
                    dynamic_scale=self.dynamic_scale if scale is None else False,
                )
                t = torch.from_numpy(patch.transpose((2, 0, 1))).float()
                tensors.append(t)
                valid_indices.append(i)
            except Exception:
                results[i] = SpoofResult(is_real=False, score=0.0, method="error")

        if tensors:
            batch_t = torch.stack(tensors).to(self.device)
            logits = self.model(batch_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for idx, prob in zip(valid_indices, probs):
                r_score = float(prob[1])
                results[idx] = SpoofResult(
                    is_real=bool(r_score >= self.threshold),
                    score=r_score,
                    method="minifasnetv2",
                )

        return results


# Backward compatibility aliases:
AntiSpoofChecker = MiniFASNetV2Checker
DeepFaceAntiSpoofChecker = MiniFASNetV2Checker
