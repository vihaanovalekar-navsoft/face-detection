"""Face Engine: Real-Time Browser-Webcam Biometric Recognition & Enrollment.

Features:
- Real-time continuous live video streaming directly from the web browser via WebRTC.
- Only bounding boxes and status labels are rendered on video (no facial points shown).
- Screen layout is completely static and never shifts.
- Single Enroll Mode toggle:
    - OFF (Default): Normal live inference. Unknown faces are labeled Unknown.
    - ON: If an unknown user is detected and you try to enroll them, a modal popup opens to input their name.
    - 16-Frame 5-Stage Multi-Angle Enrollment (4 Front, 3 Left, 3 Right, 3 Up, 3 Down) with visual progress bars.
- Internal calibrated settings: zero user-facing sliders or configuration knobs.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

# Launch via Streamlit CLI if run directly with python
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None and __name__ == "__main__":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:])
        sys.exit(0)
except Exception:
    pass

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from face_engine import config
from face_engine.detect import load_detector, detect
from face_engine.align import FaceAligner
from face_engine.embed import FaceEmbedder
from face_engine.gallery import Gallery
from face_engine.pose import estimate_head_pose, MultiAngleProfileManager
from face_engine.antispoof import MiniFASNetV2Checker
from face_engine.download_models import ensure_models_exist

# WebRTC STUN Configuration for Cloud & Local NAT Traversal
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

# Page Configuration
st.set_page_config(
    page_title="Face Engine",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS: Hide pause buttons, camera toggles, and video player controls
st.markdown(
    """
    <style>
    /* Hide video controls bar, pause, and camera toggle buttons */
    video::-webkit-media-controls {
        display: none !important;
    }
    div[data-testid="stWebRtc"] button[title*="Pause"],
    div[data-testid="stWebRtc"] button[title*="Turn off"],
    div[data-testid="stWebRtc"] button[title*="camera"],
    div[data-testid="stWebRtc"] button[aria-label*="Pause"],
    div[data-testid="stWebRtc"] button[aria-label*="camera"],
    div[data-testid="stWebRtc"] .media-controls {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load and cache deep learning models (YOLO11, SCRFD, ArcFace, MiniFASNetV2)."""
    ensure_models_exist()
    detector = load_detector(config.YOLO_WEIGHTS)
    aligner = FaceAligner(config.SCRFD_WEIGHTS, det_size=config.SCRFD_DET_SIZE)
    embedder = FaceEmbedder(config.ARCFACE_WEIGHTS)
    checker = MiniFASNetV2Checker(weights_path=config.MINIFASNET_WEIGHTS, threshold=config.SPOOF_THRESHOLD)
    return detector, aligner, embedder, checker


def get_gallery() -> Gallery:
    """Load master template gallery from disk."""
    return Gallery(config.GALLERY_PATH)


def get_next_available_name(gallery_ids: list[str]) -> str:
    n = 1
    while f"User_{n}" in gallery_ids:
        n += 1
    return f"User_{n}"


class RealtimeBiometricProcessor(VideoProcessorBase):
    """Process continuous WebRTC video frames. Strictly renders bounding boxes only (no keypoint dots)."""

    def __init__(self, detector, aligner, embedder, checker, gallery):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.checker = checker
        self.gallery = gallery
        self.lock = threading.Lock()
        self.latest_face: dict | None = None
        self.latest_unknown_face: dict | None = None
        self.active_enroll_manager: MultiAngleProfileManager | None = None
        self.completed_enroll_info: dict | None = None

    def update_gallery(self, gallery: Gallery):
        with self.lock:
            self.gallery = gallery

    def start_enrollment(self, name: str):
        with self.lock:
            self.active_enroll_manager = MultiAngleProfileManager(name, config.ENROLLED_DIR)
            self.completed_enroll_info = None

    def cancel_enrollment(self):
        with self.lock:
            self.active_enroll_manager = None

    def clear_completed_enrollment(self):
        with self.lock:
            self.completed_enroll_info = None

    def get_enroll_status(self) -> dict | None:
        with self.lock:
            if self.active_enroll_manager is not None:
                return self.active_enroll_manager.get_status_dict()
            return None

    def get_completed_enrollment(self) -> dict | None:
        with self.lock:
            return self.completed_enroll_info.copy() if self.completed_enroll_info else None

    def get_state(self) -> tuple[dict | None, dict | None]:
        with self.lock:
            lf = self.latest_face.copy() if self.latest_face else None
            lu = self.latest_unknown_face.copy() if self.latest_unknown_face else None
            return lf, lu

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        with self.lock:
            cur_gallery = self.gallery
            enroll_mgr = self.active_enroll_manager

        # 1. YOLO11 Face Detection
        boxes, confs = detect(self.detector, img, conf=config.YOLO_CONF)

        cur_primary_face = None
        cur_unknown_face = None

        if len(boxes) > 0:
            # Sort largest face first
            sorted_indices = sorted(
                range(len(boxes)),
                key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
                reverse=True,
            )

            for rank, i in enumerate(sorted_indices):
                bx = tuple(map(int, boxes[i]))
                fx1, fy1, fx2, fy2 = bx

                # Clamp box to image bounds
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(w - 1, fx2), min(h - 1, fy2)
                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                # 2. SCRFD 5-point Alignment (internal; dots NOT drawn on screen)
                aligned_crop, kps = self.aligner.align_face_with_kps(img, (fx1, fy1, fx2, fy2))

                # 3. 3D Head Pose
                pose = estimate_head_pose(kps)

                # 4. ArcFace 512-d Embedding
                emb = self.embedder.embed(aligned_crop)

                # 5. MiniFASNetV2 Anti-Spoof Evaluation
                s_res = self.checker.check(img, (fx1, fy1, fx2, fy2))
                is_spoof = not s_res.is_real
                spoof_score = s_res.score

                # Enrollment mode on primary face
                if rank == 0 and enroll_mgr is not None and not enroll_mgr.is_complete:
                    if is_spoof:
                        box_col = (0, 0, 230)  # Red
                        status_text = f"SPOOF ATTACK ({spoof_score:.2f}) [REJECTED]"
                        prompt_text = "Live human face required for enrollment"
                    else:
                        captured, msg = enroll_mgr.try_capture(
                            kps, img, (fx1, fy1, fx2, fy2), aligned_crop, emb, stability_threshold=0.50
                        )

                        if enroll_mgr.is_complete:
                            # 16 frames collected! Save to disk and register in gallery
                            enroll_mgr.save_to_disk()
                            all_embs = [f.embedding for f in enroll_mgr.frames]
                            cur_gallery.enroll_profile(enroll_mgr.name, all_embs)
                            cur_gallery.save(config.GALLERY_PATH)
                            with self.lock:
                                self.completed_enroll_info = {
                                    "name": enroll_mgr.name,
                                    "montage": str(enroll_mgr.output_dir / "montage.jpg"),
                                    "total": 16,
                                }
                                self.active_enroll_manager = None
                            box_col = (0, 255, 120)  # Green
                            status_text = f"ENROLLMENT COMPLETE ({enroll_mgr.name.upper()})"
                            prompt_text = "All 16 frames captured successfully!"
                        else:
                            box_col = (0, 215, 255)  # Vibrant Gold
                            cur_st = enroll_mgr.current_stage
                            status_text = f"ENROLLING {enroll_mgr.name.upper()} | {cur_st['name']} ({enroll_mgr.stage_collected}/{cur_st['required']}) [{enroll_mgr.total_collected}/16]"
                            prompt_text = f"👉 {cur_st['desc']}"

                    # Draw Bounding Box & Corner Accents (NO dots on face)
                    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), box_col, 2)
                    fw, fh = fx2 - fx1, fy2 - fy1
                    c_len = min(18, max(6, fw // 6), max(6, fh // 6))
                    for cx, cy in [(fx1, fy1), (fx2, fy1), (fx1, fy2), (fx2, fy2)]:
                        sx = 1 if cx == fx1 else -1
                        sy = 1 if cy == fy1 else -1
                        cv2.line(img, (cx, cy), (cx + sx * c_len, cy), box_col, 3)
                        cv2.line(img, (cx, cy), (cx, cy + sy * c_len), box_col, 3)

                    # Top Banner
                    lbl_sz = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)[0]
                    lbl_y1 = max(24, fy1 - 10)
                    lbl_y2 = lbl_y1 + lbl_sz[1] + 8
                    cv2.rectangle(img, (fx1, lbl_y1 - 6), (fx1 + lbl_sz[0] + 12, lbl_y2), (20, 20, 20), -1)
                    cv2.rectangle(img, (fx1, lbl_y1 - 6), (fx1 + lbl_sz[0] + 12, lbl_y2), box_col, 1)
                    cv2.putText(img, status_text, (fx1 + 6, lbl_y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

                    # Sub-banner prompt below bounding box
                    p_sz = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)[0]
                    p_y1 = min(h - 12, fy2 + 10)
                    p_y2 = min(h - 2, p_y1 + p_sz[1] + 6)
                    cv2.rectangle(img, (fx1, p_y1), (fx1 + p_sz[0] + 12, p_y2), (25, 25, 25), -1)
                    cv2.rectangle(img, (fx1, p_y1), (fx1 + p_sz[0] + 12, p_y2), box_col, 1)
                    cv2.putText(img, prompt_text, (fx1 + 6, p_y2 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                    cur_primary_face = {
                        "matched_id": None,
                        "similarity": 0.0,
                        "is_spoof": is_spoof,
                        "spoof_score": spoof_score,
                        "pose": pose,
                        "embedding": emb,
                        "crop": aligned_crop,
                        "timestamp": time.time(),
                    }

                else:
                    # Normal biometric recognition mode
                    matched_id = None
                    matched_score = 0.0
                    if len(cur_gallery.ids) > 0:
                        matched_id, matched_score = cur_gallery.match(emb, threshold=config.SAME_PERSON_THRESHOLD)

                    # Visual HUD Styling: ONLY bounding boxes and labels (NO face points)
                    if is_spoof:
                        box_col = (0, 0, 230)  # Red
                        status_text = f"SPOOF ATTACK ({spoof_score:.2f}) [REJECTED]"
                    elif matched_id:
                        box_col = (40, 220, 40)  # Emerald Green
                        status_text = f"{matched_id.upper()} ({matched_score:.2f}) [Live: {spoof_score:.2f}]"
                    else:
                        box_col = (0, 160, 255)  # Orange
                        status_text = f"UNKNOWN PERSON [Live: {spoof_score:.2f}]"

                    # Draw ONLY Bounding Box & Corner Accents (NO dots on face)
                    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), box_col, 2)
                    fw, fh = fx2 - fx1, fy2 - fy1
                    c_len = min(18, max(6, fw // 6), max(6, fh // 6))
                    for cx, cy in [(fx1, fy1), (fx2, fy1), (fx1, fy2), (fx2, fy2)]:
                        sx = 1 if cx == fx1 else -1
                        sy = 1 if cy == fy1 else -1
                        cv2.line(img, (cx, cy), (cx + sx * c_len, cy), box_col, 3)
                        cv2.line(img, (cx, cy), (cx, cy + sy * c_len), box_col, 3)

                    # Label Banner (NO 5 points drawn on face)
                    lbl_sz = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)[0]
                    lbl_y1 = max(24, fy1 - 10)
                    lbl_y2 = lbl_y1 + lbl_sz[1] + 8
                    cv2.rectangle(img, (fx1, lbl_y1 - 6), (fx1 + lbl_sz[0] + 12, lbl_y2), (20, 20, 20), -1)
                    cv2.rectangle(img, (fx1, lbl_y1 - 6), (fx1 + lbl_sz[0] + 12, lbl_y2), box_col, 1)
                    cv2.putText(img, status_text, (fx1 + 6, lbl_y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

                    face_info = {
                        "matched_id": matched_id,
                        "similarity": matched_score,
                        "is_spoof": is_spoof,
                        "spoof_score": spoof_score,
                        "pose": pose,
                        "embedding": emb,
                        "crop": aligned_crop,
                        "timestamp": time.time(),
                    }

                    if rank == 0:
                        cur_primary_face = face_info

                    if matched_id is None and not is_spoof and cur_unknown_face is None:
                        cur_unknown_face = face_info

        with self.lock:
            self.latest_face = cur_primary_face
            self.latest_unknown_face = cur_unknown_face

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# MODAL POPUP DIALOG FOR ENROLLMENT
# ==============================================================================
@st.dialog("Enroll Unknown Person")
def enroll_dialog(unknown_face: dict, processor: RealtimeBiometricProcessor):
    st.write("An unknown person was detected. Enter a name to capture all 16 multi-angle biometric frames:")
    col_crop, col_meta = st.columns([1, 2])
    with col_crop:
        st.image(cv2.cvtColor(unknown_face["crop"], cv2.COLOR_BGR2RGB), width=100, caption="Live Crop")
    with col_meta:
        st.write(f"- **Liveness Score:** `{unknown_face['spoof_score']:.2f}` (Verified Real)")
        st.write(f"- **Head Pose:** `{unknown_face['pose'].label}`")

    gallery = get_gallery()
    default_name = get_next_available_name(gallery.ids)
    name_input = st.text_input("Person Name / ID:", placeholder=default_name, key="enroll_dialog_name")
    target_name = name_input.strip() or default_name

    st.caption("Enrollment captures 16 frames across 5 poses: 4 Frontal + 3 Left + 3 Right + 3 Up + 3 Down.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start 16-Frame Capture", type="primary", use_container_width=True):
            clean_name = re.sub(r'[\\/*?:"<>|]', "", target_name).strip()
            if not clean_name:
                st.error("Please enter a valid identity name.")
            else:
                processor.start_enrollment(clean_name)
                st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ==============================================================================
# MAIN INTERFACE (LAYOUT NEVER SHIFTS)
# ==============================================================================
detector, aligner, embedder, checker = load_models()
gallery = get_gallery()

# Header Bar: Title + Single Specific Enrollment Toggle
col_title, col_toggle = st.columns([3, 1])
with col_title:
    st.title("Face Engine")

with col_toggle:
    st.write("")
    enroll_mode = st.toggle(
        "➕ **Enroll Mode**",
        value=False,
        key="enroll_mode",
        help="When enabled, allows unknown people detected in the camera stream to be enrolled into the gallery.",
    )

# Continuous Live Layout (Columns stay completely identical at all times)
col_stream, col_side = st.columns([3, 2])

with col_stream:
    webrtc_ctx = webrtc_streamer(
        key="live-biometric-feed",
        video_processor_factory=lambda: RealtimeBiometricProcessor(
            detector, aligner, embedder, checker, gallery
        ),
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        video_html_attrs={"autoPlay": True, "controls": False, "style": {"width": "100%"}},
        media_toggle_controls=False,
        async_processing=True,
    )


# ==============================================================================
# LIVE STATUS & 16-FRAME MULTI-ANGLE ENROLLMENT SIDE PANEL
# ==============================================================================
with col_side:
    @st.fragment(run_every=0.4)
    def render_side_panel():
        if not webrtc_ctx.state.playing or webrtc_ctx.video_processor is None:
            st.subheader("Live Status")
            st.info("Click **'START'** to enable camera stream.")
            return

        proc: RealtimeBiometricProcessor = webrtc_ctx.video_processor

        # ----------------------------------------------------------------------
        # Case A: Completed Enrollment (Celebration & Montage)
        # ----------------------------------------------------------------------
        completed = proc.get_completed_enrollment()
        if completed:
            st.subheader("Enrollment Complete")
            st.success(f"🎉 Successfully enrolled **{completed['name']}** with all 16 multi-angle frames!")
            montage_path = Path(completed.get("montage", ""))
            if montage_path.exists():
                montage_img = cv2.imread(str(montage_path))
                if montage_img is not None:
                    st.image(
                        cv2.cvtColor(montage_img, cv2.COLOR_BGR2RGB),
                        caption=f"16-Frame 3D Biometric Profile: {completed['name']}",
                        use_container_width=True,
                    )
            if st.button("✅ Done / Continue", type="primary", use_container_width=True):
                proc.clear_completed_enrollment()
                st.rerun()
            return

        # ----------------------------------------------------------------------
        # Case B: Actively Capturing 16-Frame Multi-Angle Profile
        # ----------------------------------------------------------------------
        enroll_status = proc.get_enroll_status()
        if enroll_status:
            st.subheader(f"🎯 Enrolling: {enroll_status['name']}")
            st.info(f"### 👉 {enroll_status['current_stage_desc']}")

            if enroll_status.get("last_message"):
                st.caption(f"Status: {enroll_status['last_message']}")

            # Overall Progress Bar
            total_c = enroll_status["total_collected"]
            pct = min(1.0, total_c / 16.0)
            st.progress(pct, text=f"Overall Progress: {total_c} / 16 Frames ({int(pct * 100)}%)")

            # 5 Canonical Pose Progress Bars
            counts = enroll_status["stage_counts"]
            st.markdown("**Pose Capture Bars:**")
            st.progress(counts["FRONT"] / 4.0, text=f"Frontal: {counts['FRONT']} / 4")
            st.progress(counts["LEFT"] / 3.0, text=f"Left Angle: {counts['LEFT']} / 3")
            st.progress(counts["RIGHT"] / 3.0, text=f"Right Angle: {counts['RIGHT']} / 3")
            st.progress(counts["UP"] / 3.0, text=f"Upward Tilt: {counts['UP']} / 3")
            st.progress(counts["DOWN"] / 3.0, text=f"Downward Tilt: {counts['DOWN']} / 3")

            # Live Telemetry
            latest_f, _ = proc.get_state()
            if latest_f and latest_f.get("pose"):
                m1, m2 = st.columns(2)
                m1.metric("Current Pose", latest_f["pose"].label)
                m2.metric("Yaw / Pitch", f"{latest_f['pose'].yaw:+.2f} / {latest_f['pose'].pitch:.2f}")

            st.write("")
            if st.button("❌ Cancel Enrollment", use_container_width=True):
                proc.cancel_enrollment()
                st.rerun()
            return

        # ----------------------------------------------------------------------
        # Case C: Normal Identification Mode
        # ----------------------------------------------------------------------
        st.subheader("Live Status")
        latest_face, unknown_face = proc.get_state()

        if latest_face is None:
            st.caption("Awaiting face in camera view...")
        else:
            if latest_face["is_spoof"]:
                st.error(f"❌ Spoof Attack Blocked (Score: {latest_face['spoof_score']:.2f})")
            elif latest_face["matched_id"]:
                st.success(f"Verified: **{latest_face['matched_id'].upper()}** (Score: {latest_face['similarity']:.2f})")
            else:
                st.warning("Unknown Person Detected")

            m1, m2 = st.columns(2)
            m1.metric("Liveness", f"{latest_face['spoof_score']:.2f}")
            m2.metric("Pose", latest_face["pose"].label)

        # Enrollment trigger: only when Enroll Mode is enabled and unknown face detected
        if enroll_mode:
            st.markdown("---")
            if unknown_face is not None:
                if st.button("➕ Enroll Unknown Face", type="primary", use_container_width=True):
                    enroll_dialog(unknown_face, proc)
            else:
                st.info("Present an unknown face to the camera to enroll.")

    render_side_panel()


# ==============================================================================
# ENROLLED GALLERY (COLLAPSIBLE, AT BOTTOM)
# ==============================================================================
st.markdown("---")
# Refresh gallery reference from disk
gallery = get_gallery()
with st.expander(f"📁 Enrolled Identities ({len(gallery.ids)} registered)", expanded=False):
    if not gallery.ids:
        st.info("No identities in the gallery yet.")
    else:
        for idx, person in enumerate(gallery.ids):
            st.markdown(f"**👤 {person}** — `{gallery.counts[idx]}` template(s)")
            p_dir = config.ENROLLED_DIR / person
            montage_p = p_dir / "montage.jpg"

            # If 4x4 montage exists, display it!
            if montage_p.exists():
                m_img = cv2.imread(str(montage_p))
                if m_img is not None:
                    st.image(
                        cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB),
                        caption=f"16-Frame Multi-Angle Profile: {person}",
                        use_container_width=True,
                    )
            else:
                img_files = sorted(p_dir.glob("*.jpg")) if p_dir.exists() else []
                if img_files:
                    thumb_cols = st.columns(min(8, len(img_files)))
                    for i, img_path in enumerate(img_files[:16]):
                        c_col = thumb_cols[i % len(thumb_cols)]
                        with c_col:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=img_path.stem, use_container_width=True)

            col_ren, col_del = st.columns([3, 1])
            with col_ren:
                new_name = st.text_input(f"Rename '{person}' to:", key=f"ren_input_{person}")
                if st.button("Apply Rename", key=f"btn_ren_{person}"):
                    c_new = re.sub(r'[\\/*?:"<>|]', "", new_name).strip()
                    if c_new and c_new != person:
                        gallery.rename(person, c_new)
                        new_dir = config.ENROLLED_DIR / c_new
                        if p_dir.exists() and not new_dir.exists():
                            p_dir.rename(new_dir)
                        st.success(f"Renamed '{person}' to '{c_new}'!")
                        st.rerun()

            with col_del:
                st.write("")
                st.write("")
                if st.button(f"Delete {person}", key=f"btn_del_{person}", type="secondary"):
                    gallery.delete(person)
                    if p_dir.exists():
                        shutil.rmtree(p_dir, ignore_errors=True)
                    st.warning(f"Deleted '{person}' from gallery.")
                    st.rerun()
            st.markdown("---")
