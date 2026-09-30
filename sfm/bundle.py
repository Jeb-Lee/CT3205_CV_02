"""Sparse bundle adjustment: jointly refine camera poses and 3D points by minimising reprojection error.

    min_{R_i, t_i, X_j}  sum_ij  rho( || pi(K, R_i X_j + t_i) - x_ij ||^2 )

Intrinsics are fixed (images were undistorted). The first camera is held fixed to remove the
gauge freedom (the global scale stays free but is harmless).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _rotate(points, rvecs):
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(theta > 1e-12, rvecs / theta, 0.0)
    c, s = np.cos(theta), np.sin(theta)
    return points * c + np.cross(k, points) * s + k * np.sum(k * points, axis=1, keepdims=True) * (1 - c)


def _residuals(params, n_cams, n_pts, cam_idx, pt_idx, obs, K, fixed_cam):
    cams = params[:6 * n_cams].reshape(n_cams, 6)
    pts = params[6 * n_cams:].reshape(n_pts, 3)
    Xc = _rotate(pts[pt_idx], cams[cam_idx, :3]) + cams[cam_idx, 3:]
    z = Xc[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = Xc[:, :2] / z * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return (uv - obs).ravel()


def bundle_adjust(K, cams: dict, points: dict, observations: list, fixed=None, max_nfev=25, loss_scale=1.5):
    """cams: img -> (R, t); points: pid -> xyz; observations: [(img, pid, (u, v)), ...].

    Returns updated (cams, points) and (rmse_before, rmse_after) in pixels.
    """
    import cv2
    img_ids = sorted(cams)
    pids = sorted(points)
    ci = {im: k for k, im in enumerate(img_ids)}
    pi = {p: k for k, p in enumerate(pids)}
    obs_l = [(ci[i], pi[p], uv) for i, p, uv in observations if i in ci and p in pi]
    cam_idx = np.array([o[0] for o in obs_l])
    pt_idx = np.array([o[1] for o in obs_l])
    obs = np.array([o[2] for o in obs_l], dtype=float)
    n_c, n_p = len(img_ids), len(pids)

    x0 = np.empty(6 * n_c + 3 * n_p)
    for im, k in ci.items():
        R, t = cams[im]
        x0[6 * k:6 * k + 3] = cv2.Rodrigues(R)[0].ravel()
        x0[6 * k + 3:6 * k + 6] = t
    x0[6 * n_c:] = np.array([points[p] for p in pids]).ravel()

    fixed = fixed if fixed is not None else img_ids[0]
    fk = ci[fixed]
    free_cam = np.array([k != fk for k in range(n_c)])
    free_mask = np.concatenate([np.repeat(free_cam, 6), np.ones(3 * n_p, bool)])

    def fun(xfree):
        x = x0.copy()
        x[free_mask] = xfree
        return _residuals(x, n_c, n_p, cam_idx, pt_idx, obs, K, fk)

    m = 2 * len(obs)
    A = lil_matrix((m, 6 * n_c + 3 * n_p), dtype=int)
    rows = np.arange(len(obs))
    for s in range(6):
        A[2 * rows, 6 * cam_idx + s] = 1
        A[2 * rows + 1, 6 * cam_idx + s] = 1
    for s in range(3):
        A[2 * rows, 6 * n_c + 3 * pt_idx + s] = 1
        A[2 * rows + 1, 6 * n_c + 3 * pt_idx + s] = 1
    A = A.tocsc()[:, free_mask]

    r0 = fun(x0[free_mask])
    sol = least_squares(fun, x0[free_mask], jac_sparsity=A, method="trf", loss="huber",
                        f_scale=loss_scale, x_scale="jac", max_nfev=max_nfev, verbose=0)
    x = x0.copy()
    x[free_mask] = sol.x
    r1 = _residuals(x, n_c, n_p, cam_idx, pt_idx, obs, K, fk)

    new_cams = {}
    for im, k in ci.items():
        new_cams[im] = (cv2.Rodrigues(x[6 * k:6 * k + 3])[0], x[6 * k + 3:6 * k + 6].copy())
    xp = x[6 * n_c:].reshape(n_p, 3)
    new_pts = {p: xp[pi[p]].copy() for p in pids}
    rmse = lambda r: float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2)) if len(r) else 0.0
    # rmse of the 2D euclidean error
    return new_cams, new_pts, (rmse(r0), rmse(r1))
