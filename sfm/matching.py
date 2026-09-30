"""Feature matching and geometric verification (Assignment 1.2).

Filtering cascade for every image pair:
    raw nearest-neighbour  ->  Lowe ratio test  ->  mutual (cross-check)  ->  RANSAC on E
"""
from __future__ import annotations

import cv2
import numpy as np


def raw_nn(d1, d2, norm):
    """Unfiltered 1-NN matches (every keypoint of image 1 gets a partner). 'Before filtering'."""
    bf = cv2.BFMatcher(norm)
    m = bf.match(d1, d2)
    return np.array([x.queryIdx for x in m]), np.array([x.trainIdx for x in m])


def match_pair(d1, d2, norm, ratio=0.8, cross_check=True):
    """Ratio test (+ optional mutual check). Returns idx1, idx2 and a small stats dict."""
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(d1, d2, k=2)
    i1, i2 = [], []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            i1.append(pair[0].queryIdx)
            i2.append(pair[0].trainIdx)
    i1, i2 = np.array(i1, dtype=int), np.array(i2, dtype=int)
    stats = dict(n_raw=len(d1), n_ratio=len(i1))
    if cross_check and len(i1):
        back = bf.match(d2, d1)
        best_back = np.full(len(d2), -1)
        for m in back:
            best_back[m.queryIdx] = m.trainIdx
        keep = best_back[i2] == i1
        i1, i2 = i1[keep], i2[keep]
    stats["n_mutual"] = len(i1)
    return i1, i2, stats


def verify_essential(p1, p2, K, thr_px=1.0, prob=0.999):
    """Geometric verification: RANSAC estimation of the essential matrix.

    Returns E, inlier mask (bool). thr_px is the max epipolar distance (pixels) for an inlier.
    """
    if len(p1) < 8:
        return None, np.zeros(len(p1), bool)
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=prob, threshold=thr_px)
    if E is None or mask is None:
        return None, np.zeros(len(p1), bool)
    if E.shape[0] > 3:      # OpenCV may stack several solutions; keep the first
        E = E[:3]
    return E, mask.ravel().astype(bool)
