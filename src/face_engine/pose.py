"""Facial Landmark Head Pose Estimation and Multi-Angle Profiling (Yaw + Pitch 3D Profiling)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


class PoseEstimate(NamedTuple):
    label: str           # "FRONT", "LEFT", "RIGHT", "UP", "DOWN", "TRANSITION"
    yaw: float           # Horizontal rotation ratio: >= +0.16 is Left (user turning left), <= -0.16 is Right
    pitch: float         # Vertical tilt ratio: <= 0.44 is Up, >= 0.67 is Down (~0.56 is neutral)
    roll: float          # In-plane tilt angle in degrees


def estimate_head_pose(kps: np.ndarray) -> PoseEstimate:
    """Estimate 3D head pose orientation from 5 SCRFD facial keypoints.
    
    kps: (5, 2) array of coordinates:
        0: left eye (viewer's left / image left)
        1: right eye (viewer's right / image right)
        2: nose tip
        3: left mouth corner
        4: right mouth corner
    """
    if kps is None or len(kps) < 5:
        return PoseEstimate(label="UNKNOWN", yaw=0.0, pitch=0.56, roll=0.0)

    eye_l = kps[0]
    eye_r = kps[1]
    nose = kps[2]
    mouth_l = kps[3]
    mouth_r = kps[4]

    # Eye center and distance
    eye_mid_x = (eye_l[0] + eye_r[0]) / 2.0
    eye_mid_y = (eye_l[1] + eye_r[1]) / 2.0
    eye_dist = max(1.0, float(np.linalg.norm(eye_r - eye_l)))

    # Yaw: horizontal displacement of nose relative to eye midpoint normalized by eye distance
    # Neutral is ~0.00 to 0.04.
    # Deliberate head rotation (from user perspective facing camera):
    # When user turns head to their LEFT, in camera image nose moves right (+yaw >= +0.16)
    # When user turns head to their RIGHT, in camera image nose moves left (-yaw <= -0.16)
    yaw = float((nose[0] - eye_mid_x) / eye_dist)

    # Pitch: vertical displacement of nose relative to eye midpoint vs mouth midpoint
    # Neutral is ~0.54 - 0.58
    # UP: pitch <= 0.44 (chin raised, nose closer to eye line)
    # DOWN: pitch >= 0.67 (chin lowered, nose closer to mouth line)
    mouth_mid_y = (mouth_l[1] + mouth_r[1]) / 2.0
    vertical_span = max(1.0, mouth_mid_y - eye_mid_y)
    pitch = float((nose[1] - eye_mid_y) / vertical_span)

    # Roll: eye tilt angle in degrees
    d_y = eye_r[1] - eye_l[1]
    d_x = eye_r[0] - eye_l[0]
    roll = float(np.degrees(np.arctan2(d_y, d_x)))

    # Classify pose with clear separation (user-centric)
    if yaw >= 0.16:
        label = "LEFT"
    elif yaw <= -0.16:
        label = "RIGHT"
    elif pitch <= 0.44:
        label = "UP"
    elif pitch >= 0.67:
        label = "DOWN"
    elif abs(yaw) <= 0.09 and 0.46 <= pitch <= 0.65:
        label = "FRONT"
    else:
        label = "TRANSITION"

    return PoseEstimate(label=label, yaw=yaw, pitch=pitch, roll=roll)


class EnrollmentFrame:
    """A captured multi-angle enrollment frame."""
    def __init__(
        self,
        index: int,
        stage: str,
        pose: PoseEstimate,
        full_frame: np.ndarray,
        face_crop: np.ndarray,
        aligned_crop: np.ndarray,
        embedding: np.ndarray,
    ):
        self.index = index
        self.stage = stage
        self.pose = pose
        self.full_frame = full_frame
        self.face_crop = face_crop
        self.aligned_crop = aligned_crop
        self.embedding = embedding


class MultiAngleProfileManager:
    """Manages 5-stage 3D multi-angle enrollment: 4 Frontal + 3 Left + 3 Right + 3 Up + 3 Down = 16 frames."""

    STAGES = [
        {"name": "FRONT", "desc": "Look STRAIGHT at camera", "required": 4},
        {"name": "LEFT",  "desc": "Turn head clearly LEFT",  "required": 3},
        {"name": "RIGHT", "desc": "Turn head clearly RIGHT", "required": 3},
        {"name": "UP",    "desc": "Tilt head slightly UP",    "required": 3},
        {"name": "DOWN",  "desc": "Tilt head slightly DOWN",  "required": 3},
    ]
    TOTAL_FRAMES = 16

    def __init__(self, name: str, output_base_dir: Path | str):
        self.name = name
        self.output_dir = Path(output_base_dir) / name
        self.frames: list[EnrollmentFrame] = []
        self.current_stage_idx = 0
        self.last_capture_time = 0.0
        self.min_interval = 0.20  # Minimum seconds between captures
        self._last_message = "Ready to capture"

    @property
    def current_stage(self) -> dict:
        if self.current_stage_idx < len(self.STAGES):
            return self.STAGES[self.current_stage_idx]
        return {"name": "COMPLETE", "desc": "Enrollment Complete", "required": 0}

    @property
    def stage_collected(self) -> int:
        cur_name = self.current_stage["name"]
        return sum(1 for f in self.frames if f.stage == cur_name)

    @property
    def total_collected(self) -> int:
        return len(self.frames)

    @property
    def is_complete(self) -> bool:
        return len(self.frames) >= self.TOTAL_FRAMES

    def get_stage_counts(self) -> dict[str, int]:
        counts = {s["name"]: 0 for s in self.STAGES}
        for f in self.frames:
            if f.stage in counts:
                counts[f.stage] += 1
        return counts

    def get_status_dict(self) -> dict:
        cur = self.current_stage
        return {
            "name": self.name,
            "is_complete": self.is_complete,
            "total_collected": self.total_collected,
            "total_required": self.TOTAL_FRAMES,
            "current_stage_name": cur["name"],
            "current_stage_desc": cur["desc"],
            "current_stage_req": cur["required"],
            "current_stage_collected": self.stage_collected,
            "stage_counts": self.get_stage_counts(),
            "last_message": getattr(self, "_last_message", ""),
        }

    def try_capture(
        self,
        kps: np.ndarray,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        aligned: np.ndarray,
        embedding: np.ndarray,
        stability_threshold: float = 0.50,
    ) -> tuple[bool, str]:
        """Check if current observation satisfies active stage requirement."""
        if self.is_complete:
            self._last_message = "Enrollment already complete"
            return False, self._last_message

        now = time.time()
        if (now - self.last_capture_time) < self.min_interval:
            return False, "Waiting interval"

        pose = estimate_head_pose(kps)
        cur_req_stage = self.current_stage["name"]

        if pose.label != cur_req_stage:
            msg = f"Need {cur_req_stage} (Current: {pose.label}, yaw={pose.yaw:+.2f}, pitch={pose.pitch:.2f})"
            self._last_message = msg
            return False, msg

        # Gating: stability check with previous frame of same stage to prevent motion blur
        stage_frames = [f for f in self.frames if f.stage == cur_req_stage]
        if stage_frames:
            sim = float(stage_frames[-1].embedding @ embedding)
            if sim < stability_threshold:
                msg = f"Motion blur / unstable (sim={sim:.2f} < {stability_threshold})"
                self._last_message = msg
                return False, msg

        # Crop face with margin
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * 0.2)
        pad_y = int(bh * 0.2)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        idx = len(self.frames) + 1
        e_frame = EnrollmentFrame(
            index=idx,
            stage=cur_req_stage,
            pose=pose,
            full_frame=frame.copy(),
            face_crop=face_crop,
            aligned_crop=aligned.copy(),
            embedding=embedding.copy(),
        )
        self.frames.append(e_frame)
        self.last_capture_time = now

        # Advance stage if requirement met
        if self.stage_collected >= self.current_stage["required"]:
            self.current_stage_idx += 1

        msg = f"Captured frame #{idx}/16 ({cur_req_stage}, yaw={pose.yaw:+.2f}, pitch={pose.pitch:.2f})"
        self._last_message = msg
        return True, msg

    def save_to_disk(self) -> dict:
        """Save all 16 face images, aligned crops, summary 4x4 montage, and metadata JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir = self.output_dir / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        montage_faces = []

        for f in self.frames:
            stem = f"frame_{f.index}_{f.stage.lower()}"
            crop_path = self.output_dir / f"{stem}.jpg"
            aligned_path = aligned_dir / f"{stem}_aligned.jpg"

            cv2.imwrite(str(crop_path), f.face_crop)
            cv2.imwrite(str(aligned_path), f.aligned_crop)

            saved_files.append({
                "index": f.index,
                "stage": f.stage,
                "yaw": round(f.pose.yaw, 3),
                "pitch": round(f.pose.pitch, 3),
                "roll": round(f.pose.roll, 1),
                "face_image": str(crop_path.name),
                "aligned_image": str(aligned_path.name),
            })

            # Prepare square 150x150 thumbnail for montage
            thumb = cv2.resize(f.aligned_crop, (150, 150))
            cv2.rectangle(thumb, (0, 126), (150, 150), (25, 25, 25), -1)
            cv2.putText(
                thumb,
                f"#{f.index} {f.stage}",
                (6, 143),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )
            montage_faces.append(thumb)

        # Build 4x4 montage grid (4 rows of 4 columns = 16 thumbnails)
        if len(montage_faces) >= 16:
            row1 = np.hstack(montage_faces[0:4])
            row2 = np.hstack(montage_faces[4:8])
            row3 = np.hstack(montage_faces[8:12])
            row4 = np.hstack(montage_faces[12:16])
            montage_grid = np.vstack([row1, row2, row3, row4])

            # Header banner
            header = np.zeros((65, montage_grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                header,
                f"ENROLLED IDENTITY: {self.name.upper()} (16-FRAME 3D PROFILE)",
                (16, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )
            date_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                header,
                f"Coverage: 4 Front + 3 Left + 3 Right + 3 Up + 3 Down | {date_str}",
                (16, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            montage_full = np.vstack([header, montage_grid])
            montage_path = self.output_dir / "montage.jpg"
            cv2.imwrite(str(montage_path), montage_full)

        # Metadata JSON
        montage_path = self.output_dir / "montage.jpg"
        meta = {
            "name": self.name,
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames": len(self.frames),
            "stages": self.STAGES,
            "frames": saved_files,
            "montage_path": str(montage_path) if montage_path.exists() else None,
        }
        meta_path = self.output_dir / "profile.json"
        with open(meta_path, "w") as fp:
            json.dump(meta, fp, indent=2)

        return meta
