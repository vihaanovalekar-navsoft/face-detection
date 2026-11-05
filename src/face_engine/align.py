"""SCRFD keypoint detection and ArcFace alignment (strictly GPU)."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from .config import SCRFD_DET_SIZE
from .onnx_utils import cuda_providers, verify_cuda_session


ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2044]],
    dtype=np.float32,
)


def _distance2bbox(points, distance):
    x1y1 = points - distance[:, :2]
    x2y2 = points + distance[:, 2:]
    return np.concatenate([x1y1, x2y2], axis=1)


def _nms(dets, scores, iou_thr=0.4):
    if len(dets) == 0:
        return []
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ovr <= iou_thr]
    return keep


class FaceAligner:
    """SCRFD detector + ArcFace 112x112 alignment strictly on CUDA GPU."""

    def __init__(self, scrfd_path, det_size=SCRFD_DET_SIZE, num_anchors=2):
        self.sess = ort.InferenceSession(str(scrfd_path), providers=cuda_providers())
        verify_cuda_session(self.sess, "SCRFD")
        self.det_size = det_size
        self.num_anchors = num_anchors

    def detect_keypoints(
        self,
        image: np.ndarray,
        thresh: float = 0.20,
        downscale: float = 1.0,
        det_size: int | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        """Return list of (kps (5,2) float32, score float)."""
        if downscale != 1.0:
            h, w = image.shape[:2]
            small = cv2.resize(image, (max(1, round(w * downscale)), max(1, round(h * downscale))))
            kps_list = self.detect_keypoints(small, thresh=thresh, downscale=1.0, det_size=det_size)
            inv = 1.0 / downscale
            return [(k * inv, s) for k, s in kps_list]

        h, w = image.shape[:2]
        target_det_size = det_size if det_size is not None else self.det_size
        scale = target_det_size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        im = cv2.resize(image, (nw, nh)).astype(np.float32)
        ph = (32 - nh % 32) % 32
        pw = (32 - nw % 32) % 32
        im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT)
        blob = ((im - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)

        outs = self.sess.run(None, {self.sess.get_inputs()[0].name: blob})
        groups = {}
        for o in outs:
            t = o.reshape(-1, o.shape[-1])
            groups.setdefault(t.shape[0], []).append(t)

        scores_all, boxes_all, kps_all = [], [], []
        ih, iw = im.shape[0], im.shape[1]
        for rows, tensors in groups.items():
            score_t = next(t for t in tensors if t.shape[1] == 1).reshape(-1)
            bbox_t = next(t for t in tensors if t.shape[1] == 4)
            kps_t = next(t for t in tensors if t.shape[1] == 10)
            if score_t.max() > 1.0 or score_t.min() < 0.0:
                score_t = 1.0 / (1.0 + np.exp(-score_t))
            stride = next(s for s in (8, 16, 32) if (ih // s) * (iw // s) * self.num_anchors == rows)
            hh, ww = ih // stride, iw // stride
            centers = np.stack(np.mgrid[:hh, :ww][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)
            boxes = _distance2bbox(centers, bbox_t * stride)
            kps = centers[:, None, :] + (kps_t * stride).reshape(-1, 5, 2)
            inds = score_t > thresh
            scores_all.append(score_t[inds])
            boxes_all.append(boxes[inds])
            kps_all.append(kps[inds])

        if not scores_all or all(len(s) == 0 for s in scores_all):
            return []
        scores = np.concatenate(scores_all)
        boxes = np.concatenate(boxes_all)
        kps = np.concatenate(kps_all)
        keep = _nms(boxes, scores)
        return [(kps[i] / scale, float(scores[i])) for i in keep]

    @staticmethod
    def align(image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Align face image to canonical 112x112 ArcFace template via partial affine similarity transform."""
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_TEMPLATE)
        if M is None:
            # Fallback if affine estimation fails
            return cv2.resize(image, (112, 112))
        return cv2.warpAffine(image, M, (112, 112), borderValue=0)

    def align_face_with_kps(
        self,
        frame: np.ndarray,
        face_box: np.ndarray,
        precomputed_kps: list[tuple[np.ndarray, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Align face image and return (aligned_crop_112x112, kps_5x2_or_None)."""
        h_frame, w_frame = frame.shape[:2]
        fx1, fy1, fx2, fy2 = face_box

        # 1. Match against precomputed full-frame landmarks
        if precomputed_kps:
            for kps, sc in precomputed_kps:
                kcx, kcy = kps[:, 0].mean(), kps[:, 1].mean()
                if (fx1 - 10 <= kcx <= fx2 + 10) and (fy1 - 10 <= kcy <= fy2 + 10):
                    return self.align(frame, kps), kps

        # 2. Local padded crop detection (lightweight det_size=192)
        fw, fh = fx2 - fx1, fy2 - fy1
        pad_x = fw * 0.40
        pad_y = fh * 0.40
        ex1 = max(0, int(fx1 - pad_x))
        ey1 = max(0, int(fy1 - pad_y))
        ex2 = min(w_frame, int(fx2 + pad_x))
        ey2 = min(h_frame, int(fy2 + pad_y))

        crop = frame[ey1:ey2, ex1:ex2]
        if crop.shape[0] >= 16 and crop.shape[1] >= 16:
            local_kps = self.detect_keypoints(crop, thresh=0.03, det_size=192)
            if local_kps:
                # Map back to frame coordinates
                best_local_kps = local_kps[0][0] + np.array([ex1, ey1], dtype=np.float32)
                return self.align(frame, best_local_kps), best_local_kps

        # 3. Context-preserving centered crop fallback with geometric landmarks
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        size = max(fw, fh) * 1.35
        half = size / 2.0
        cx1 = max(0, int(cx - half))
        cy1 = max(0, int(cy - half))
        cx2 = min(w_frame, int(cx + half))
        cy2 = min(h_frame, int(cy + half))

        approx_kps = np.array([
            [fx1 + fw * 0.32, fy1 + fh * 0.38],
            [fx1 + fw * 0.68, fy1 + fh * 0.38],
            [fx1 + fw * 0.50, fy1 + fh * 0.55],
            [fx1 + fw * 0.36, fy1 + fh * 0.74],
            [fx1 + fw * 0.64, fy1 + fh * 0.74],
        ], dtype=np.float32)

        safe_crop = frame[cy1:cy2, cx1:cx2]
        if safe_crop.shape[0] >= 10 and safe_crop.shape[1] >= 10:
            return cv2.resize(safe_crop, (112, 112), interpolation=cv2.INTER_LINEAR), approx_kps
        return cv2.resize(frame[max(0, int(fy1)):min(h_frame, int(fy2)), max(0, int(fx1)):min(w_frame, int(fx2))], (112, 112)), approx_kps

    def align_face(
        self,
        frame: np.ndarray,
        face_box: np.ndarray,
        precomputed_kps: list[tuple[np.ndarray, float]] | None = None,
    ) -> np.ndarray:
        """Intelligently align face crop:
        1. Search in precomputed full-frame keypoints.
        2. Fallback to padded local crop landmark search.
        3. Fallback to context-preserving centered crop.
        """
        crop, _ = self.align_face_with_kps(frame, face_box, precomputed_kps)
        return crop

    align_canonical_crop = align_face
