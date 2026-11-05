"""Face detection with YOLO11-face (GPU accelerated with automatic CPU fallback)."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_device() -> str:
    """Return 'cuda:0' if CUDA is available, else 'cpu'."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_detector(weights_path):
    device = get_device()
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def detect(model, image, conf=0.50, imgsz=1280):
    """Return (boxes (N,4) xyxy, confidences (N,)). GPU preferred, CPU fallback."""
    device = get_device()
    res = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
    confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,))
    return boxes, confs


def annotate(image, boxes, confs):
    out = image.copy()
    for bx, cf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, bx)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(out, f"face {cf:.2f}", (x1, max(14, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    return out
