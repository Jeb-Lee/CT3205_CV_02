"""Two-view and multi-view geometry primitives, implemented explicitly (Assignment 1.3 / 1.5).

Conventions: a camera is (R, t) with x_cam = R X + t (world -> camera). P = K [R | t].
"""
from __future__ import annotations

import cv2
import numpy as np


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def to_normalized(pts, K):
    """Pixel coords -> normalised image coords  x_n = K^-1 [u v 1]^T."""
    return (pts - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])


def decompose_essential(E):
    """E = [t]x R has four (R, t) solutions: {R1, R2} x {+t, -t}."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate_dlt(R1, t1, R2, t2, xn1, xn2):
    """Linear (DLT) triangulation of N correspondences in normalised coordinates.

    For each point solve A X = 0 with rows  x*P[2]-P[0]  and  y*P[2]-P[1]  from both views
    and take the right singular vector of the smallest singular value.
    """
    P1 = np.hstack([R1, t1.reshape(3, 1)])
    P2 = np.hstack([R2, t2.reshape(3, 1)])
    n = len(xn1)
    A = np.empty((n, 4, 4))
    A[:, 0] = xn1[:, 0:1] * P1[2] - P1[0]
    A[:, 1] = xn1[:, 1:2] * P1[2] - P1[1]
    A[:, 2] = xn2[:, 0:1] * P2[2] - P2[0]
    A[:, 3] = xn2[:, 1:2] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]
    w = X[:, 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return X[:, :3] / w


def depth(R, t, X):
    return (X @ R.T + t)[:, 2]


def project(K, R, t, X):
    """Pixel projection of world points; returns (N,2) pixels and (N,) depths."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    uv = Xc[:, :2] / zs[:, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, z


def reproj_error(K, R, t, X, pts):
    uv, z = project(K, R, t, X)
    return np.linalg.norm(uv - pts, axis=1), z


def camera_center(R, t):
    return -R.T @ t


def triangulation_angle_deg(C1, C2, X):
    """Angle subtended at X by the two camera centres - the 'parallax'. Small = poorly conditioned depth."""
    a, b = C1 - X, C2 - X
    cosang = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def recover_pose_cheirality(E, K, p1, p2):
    """Pick the physically valid (R, t) out of the four candidates of E.

    Cheirality: a reconstructed point must lie IN FRONT of both cameras (positive depth).
    Returns R, t (unit norm), per-candidate counts and the 3D points of the winner.
    """
    xn1, xn2 = to_normalized(p1, K), to_normalized(p2, K)
    counts, cands = [], decompose_essential(E)
    best = None
    for R, t in cands:
        X = triangulate_dlt(np.eye(3), np.zeros(3), R, t, xn1, xn2)
        n_ok = int(np.sum((depth(np.eye(3), np.zeros(3), X) > 0) & (depth(R, t, X) > 0)))
        counts.append(n_ok)
        if best is None or n_ok > best[0]:
            best = (n_ok, R, t, X)
    return best[1], best[2], counts, best[3]


def rotation_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def rodrigues_vec(R):
    return cv2.Rodrigues(R)[0].ravel()


def rodrigues_mat(r):
    return cv2.Rodrigues(np.asarray(r, dtype=float).reshape(3, 1))[0]


def sampson_error(F, p1, p2):
    """First-order geometric (Sampson) distance of correspondences to epipolar constraint, in px^2."""
    x1 = np.hstack([p1, np.ones((len(p1), 1))])
    x2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / (den + 1e-12)
