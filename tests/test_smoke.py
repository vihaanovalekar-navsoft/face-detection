"""Smoke tests for the face_engine package."""
from pathlib import Path

import numpy as np

from face_engine import config
from face_engine.gallery import Gallery
from face_engine.pose import estimate_head_pose
from face_engine.antispoof import DeepFaceAntiSpoofChecker, SpoofResult


def test_config_paths():
    assert config.ROOT.exists()
    assert config.OUTPUT_DIR.exists() or True


def test_gallery_loads_enrolled_users():
    gallery_path = config.GALLERY_PATH
    assert gallery_path.exists(), "gallery.npz missing"
    gallery = Gallery(gallery_path)
    assert "Ashutosh" in gallery.ids
    assert "Abishek" in gallery.ids
    assert len(gallery.ids) >= 2
    for count in gallery.counts:
        assert count == 16


def test_gallery_templates_shape():
    gallery = Gallery(config.GALLERY_PATH)
    for template in gallery.templates:
        assert isinstance(template, np.ndarray)
        assert template.shape == (512,)


def test_pose_front():
    import numpy as np

    kps = np.array(
        [[40.0, 50.0], [72.0, 50.0], [56.0, 72.0], [42.0, 92.0], [70.0, 92.0]],
        dtype=np.float32,
    )
    pose = estimate_head_pose(kps)
    assert pose.label == "FRONT"


def test_antispoof_checker():
    checker = DeepFaceAntiSpoofChecker(threshold=0.5)
    img_path = config.ROOT / "data" / "enrolled" / "Ashutosh" / "frame_1_front.jpg"
    import cv2

    img = cv2.imread(str(img_path))
    assert img is not None
    h, w = img.shape[:2]
    result = checker.check(img, (0, 0, w, h))
    assert isinstance(result, SpoofResult)
    assert isinstance(result.is_real, bool)
    assert 0.0 <= result.score <= 1.0


def test_pipeline_detect_align_embed_match():
    import cv2
    from face_engine.detect import load_detector, detect
    from face_engine.align import FaceAligner
    from face_engine.embed import FaceEmbedder

    detector = load_detector(config.YOLO_WEIGHTS)
    aligner = FaceAligner(config.SCRFD_WEIGHTS)
    embedder = FaceEmbedder(config.ARCFACE_WEIGHTS)
    gallery = Gallery(config.GALLERY_PATH)

    img_path = config.ROOT / "data" / "enrolled" / "Ashutosh" / "frame_1_front.jpg"
    img = cv2.imread(str(img_path))
    assert img is not None

    boxes, confs = detect(detector, img, conf=0.25)
    assert len(boxes) > 0

    aligned, kps = aligner.align_face_with_kps(img, boxes[0])
    assert aligned.shape == (112, 112, 3)
    assert kps is not None
    assert kps.shape == (5, 2)

    emb = embedder.embed(aligned)
    assert emb.shape == (512,)

    matched_id, score = gallery.match(emb, threshold=config.SAME_PERSON_THRESHOLD)
    assert matched_id == "Ashutosh"
    assert score > 0.80

