"""Test 16-frame 5-stage multi-angle biometric profile enrollment and gallery persistence."""
import tempfile
from pathlib import Path
import numpy as np

from face_engine.pose import MultiAngleProfileManager, estimate_head_pose
from face_engine.gallery import Gallery


def test_16_frame_multi_angle_enrollment():
    kps_front = np.array([[75, 100], [125, 100], [100, 128], [80, 150], [120, 150]], dtype=np.float32)
    kps_left = np.array([[75, 100], [125, 100], [110, 128], [80, 150], [120, 150]], dtype=np.float32)
    kps_right = np.array([[75, 100], [125, 100], [90, 128], [80, 150], [120, 150]], dtype=np.float32)
    kps_up = np.array([[75, 100], [125, 100], [100, 120], [80, 150], [120, 150]], dtype=np.float32)
    kps_down = np.array([[75, 100], [125, 100], [100, 136], [80, 150], [120, 150]], dtype=np.float32)

    assert estimate_head_pose(kps_front).label == "FRONT"
    assert estimate_head_pose(kps_left).label == "LEFT"
    assert estimate_head_pose(kps_right).label == "RIGHT"
    assert estimate_head_pose(kps_up).label == "UP"
    assert estimate_head_pose(kps_down).label == "DOWN"

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = MultiAngleProfileManager("TestSubject", tmpdir)
        mgr.min_interval = 0.0

        for _ in range(4):
            ok, _ = mgr.try_capture(
                kps_front, np.zeros((200, 200, 3), np.uint8), (50, 50, 150, 150),
                np.zeros((112, 112, 3), np.uint8), np.ones(512, np.float32) / np.sqrt(512)
            )
            assert ok

        for _ in range(3):
            ok, _ = mgr.try_capture(
                kps_left, np.zeros((200, 200, 3), np.uint8), (50, 50, 150, 150),
                np.zeros((112, 112, 3), np.uint8), np.ones(512, np.float32) / np.sqrt(512)
            )
            assert ok

        for _ in range(3):
            ok, _ = mgr.try_capture(
                kps_right, np.zeros((200, 200, 3), np.uint8), (50, 50, 150, 150),
                np.zeros((112, 112, 3), np.uint8), np.ones(512, np.float32) / np.sqrt(512)
            )
            assert ok

        for _ in range(3):
            ok, _ = mgr.try_capture(
                kps_up, np.zeros((200, 200, 3), np.uint8), (50, 50, 150, 150),
                np.zeros((112, 112, 3), np.uint8), np.ones(512, np.float32) / np.sqrt(512)
            )
            assert ok

        for _ in range(3):
            ok, _ = mgr.try_capture(
                kps_down, np.zeros((200, 200, 3), np.uint8), (50, 50, 150, 150),
                np.zeros((112, 112, 3), np.uint8), np.ones(512, np.float32) / np.sqrt(512)
            )
            assert ok

        assert mgr.is_complete
        st = mgr.get_status_dict()
        assert st["stage_counts"] == {"FRONT": 4, "LEFT": 3, "RIGHT": 3, "UP": 3, "DOWN": 3}

        meta = mgr.save_to_disk()
        assert Path(meta["montage_path"]).exists()

        gal_path = Path(tmpdir) / "gallery.npz"
        gal = Gallery(gal_path)
        embs = [f.embedding for f in mgr.frames]
        gal.enroll_profile("TestSubject", embs)

        assert "TestSubject" in gal.ids
        assert gal.counts[gal.ids.index("TestSubject")] == 16

        query = np.ones(512, np.float32) / np.sqrt(512)
        match_id, score = gal.match(query, threshold=0.70)
        assert match_id == "TestSubject"
        assert score >= 0.70

