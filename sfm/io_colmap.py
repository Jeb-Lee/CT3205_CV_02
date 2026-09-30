"""Readers for COLMAP text-format models (cameras.txt / images.txt / points3D.txt).

The COLMAP output shipped with the datasets is used ONLY as (a) the source of camera
intrinsics and (b) ground truth for evaluation. The reconstruction itself never reads poses.
"""
from __future__ import annotations

import numpy as np


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def _data_lines(path, keep_blank=False):
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return [ln for ln in lines if not ln.startswith("#") and (keep_blank or ln.strip())]


def read_cameras(path):
    cams = {}
    for ln in _data_lines(path):
        tok = ln.split()
        cams[int(tok[0])] = dict(model=tok[1], width=int(tok[2]), height=int(tok[3]),
                                 params=np.array([float(v) for v in tok[4:]]))
    return cams


def camera_to_K_dist(cam):
    """Return (K, dist) in OpenCV convention for the COLMAP camera models we meet."""
    m, p = cam["model"], cam["params"]
    if m == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), np.zeros(5)
    if m == "PINHOLE":
        fx, fy, cx, cy = p
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.zeros(5)
    if m == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), np.array([k, 0, 0, 0, 0.0])
    if m == "RADIAL":
        f, cx, cy, k1, k2 = p
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), np.array([k1, k2, 0, 0, 0.0])
    if m == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.array([k1, k2, p1, p2, 0.0])
    raise ValueError(f"unsupported COLMAP camera model {m}")


def read_images(path):
    """Return dict name -> dict(R, t, cam_id) with world->camera convention x_c = R X + t."""
    lines = _data_lines(path, keep_blank=True)
    out = {}
    i = 0
    while i + 1 < len(lines):
        tok = lines[i].split()
        if len(tok) < 10:  # stray blank line
            i += 1
            continue
        q = np.array([float(v) for v in tok[1:5]])
        t = np.array([float(v) for v in tok[5:8]])
        out[tok[9]] = dict(R=qvec2rotmat(q), t=t, cam_id=int(tok[8]))
        i += 2
    return out


def read_points3D(path):
    xyz, rgb, err = [], [], []
    for ln in _data_lines(path):
        tok = ln.split()
        xyz.append([float(v) for v in tok[1:4]])
        rgb.append([int(v) for v in tok[4:7]])
        err.append(float(tok[7]))
    return np.array(xyz), np.array(rgb, dtype=np.uint8), np.array(err)
