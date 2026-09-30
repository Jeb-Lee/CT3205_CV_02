"""Feature detection / description (Assignment 1.1)."""
from __future__ import annotations

import cv2
import numpy as np

DETECTORS = ("sift", "rootsift", "orb", "akaze", "brisk")


def _factory(cls_name: str):
    """OpenCV 5 moved AKAZE/BRISK/KAZE from `cv2` into the contrib namespace `cv2.xfeatures2d`; OpenCV 4 has them in `cv2`."""
    for ns in (cv2, getattr(cv2, "xfeatures2d", None)):
        if ns is not None and hasattr(ns, f"{cls_name}_create"):
            return getattr(ns, f"{cls_name}_create")
    raise RuntimeError(f"{cls_name} not available in this OpenCV build (need opencv-contrib-python)")


def make_detector(name: str, nfeatures: int):
    name = name.lower()
    if name in ("sift", "rootsift"):
        return cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
    if name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    if name == "akaze":
        return _factory("AKAZE")(threshold=5e-4)
    if name == "brisk":
        return _factory("BRISK")(thresh=20)
    raise ValueError(name)


def norm_for(name: str):
    return cv2.NORM_L2 if name.lower() in ("sift", "rootsift") else cv2.NORM_HAMMING


def extract(gray: np.ndarray, detector: str, nfeatures: int):
    """Return keypoints (N,2 float32 pixel coords) and descriptors (N,D)."""
    det = make_detector(detector, nfeatures)
    kps, desc = det.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), None
    if len(kps) > nfeatures:  # AKAZE/BRISK have no cap: keep strongest
        order = np.argsort([-k.response for k in kps])[:nfeatures]
        kps = [kps[i] for i in order]
        desc = desc[order]
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    if detector.lower() == "rootsift":  # Arandjelovic & Zisserman: L1-normalise then sqrt
        desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc).astype(np.float32)
    return pts, desc
