"""Quantitative evaluation against the COLMAP ground truth shipped with the datasets.

Monocular SfM is defined only up to a similarity transform (rotation, translation, scale) so we
first align our model to the ground truth with Umeyama's closed-form solution on the camera centres.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as G


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(S) @ D) / var if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_umeyama(src, dst, iters=200, thr_frac=0.05, seed=0):
    """RANSAC around Umeyama so a single badly-registered camera cannot ruin the alignment."""
    n = len(src)
    if n < 4:
        return umeyama(src, dst)
    rng = np.random.default_rng(seed)
    scale = np.linalg.norm(dst - dst.mean(0), axis=1).max() + 1e-9
    best, best_inl = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except Exception:
            continue
        err = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        inl = err < thr_frac * scale
        if inl.sum() > best_inl:
            best_inl, best = inl.sum(), inl
    if best is None or best.sum() < 3:
        return umeyama(src, dst)
    return umeyama(src[best], dst[best])


def evaluate_against_gt(rec, pc=None):
    """Return metrics dict comparing cameras and points of `rec` (Reconstruction) with COLMAP GT."""
    ds = rec.ds
    if not ds.gt_poses:
        return {}
    ids = [i for i in rec.cams if rec.names[i] in ds.gt_poses]
    if len(ids) < 3:
        return dict(gt_n_cams=len(ids))
    C_est = np.array([G.camera_center(*rec.cams[i]) for i in ids])
    C_gt = np.array([G.camera_center(*ds.gt_poses[rec.names[i]]) for i in ids])
    s, R, t = robust_umeyama(C_est, C_gt)
    # Near-collinear camera centres leave the rotation about the trajectory line unconstrained (few images on a
    # straight path). Add each camera's optical-axis point as extra correspondences to pin that rotation down.
    sv = np.linalg.svd(C_gt - C_gt.mean(0), compute_uv=False)
    collinear = sv[1] / (sv[0] + 1e-12) < 0.1
    if collinear:
        d = np.linalg.norm(C_est - C_est.mean(0), axis=1).max() + 1e-9
        ax_est = np.array([rec.cams[i][0].T @ np.array([0, 0, 1.0]) for i in ids])
        ax_gt = np.array([ds.gt_poses[rec.names[i]][0].T @ np.array([0, 0, 1.0]) for i in ids])
        s, R, t = umeyama(np.vstack([C_est, C_est + d * ax_est]), np.vstack([C_gt, C_gt + s * d * ax_gt]))
    C_al = s * C_est @ R.T + t
    pos_err = np.linalg.norm(C_al - C_gt, axis=1)
    rot_err = []
    for k, i in enumerate(ids):
        Re = rec.cams[i][0]                         # world_est -> cam
        Rg = ds.gt_poses[rec.names[i]][0]           # world_gt  -> cam
        # world_est = R^T-aligned: X_gt = s R X_est + t  =>  R_est_in_gt = Re R^T
        rot_err.append(G.rotation_angle_deg(Re @ R.T @ Rg.T))
    extent = np.linalg.norm(C_gt - C_gt.mean(0), axis=1).max() + 1e-9
    out = dict(gt_n_cams=len(ids), align_scale=float(s), align_collinear_fix=bool(collinear),
               cam_pos_rmse=float(np.sqrt((pos_err ** 2).mean())),
               cam_pos_rmse_rel=float(np.sqrt((pos_err ** 2).mean()) / extent),   # relative to trajectory extent
               cam_pos_median=float(np.median(pos_err)),
               cam_rot_mean_deg=float(np.mean(rot_err)), cam_rot_median_deg=float(np.median(rot_err)),
               cam_rot_max_deg=float(np.max(rot_err)))
    pc = pc or rec.point_cloud(with_colors=False)
    if len(pc["xyz"]) and ds.gt_points is not None:
        Xal = s * pc["xyz"] @ R.T + t
        tree_gt = cKDTree(ds.gt_points)
        d, _ = tree_gt.query(Xal)
        # accuracy: distance of our points to nearest GT point, in units of the camera-trajectory extent
        out.update(pt_acc_median=float(np.median(d) / extent), pt_acc_mean=float(np.mean(d) / extent),
                   pt_within_2pct=float(np.mean(d < 0.02 * extent)))
        tree_est = cKDTree(Xal)
        d2, _ = tree_est.query(ds.gt_points)
        out.update(gt_completeness_2pct=float(np.mean(d2 < 0.02 * extent)))
    out["_sim3"] = (s, R, t)
    return out


def match_correctness(rec, i, j, i1, i2, thr_px=1.5):
    """Label matches of pair (i, j) using GT epipolar geometry. Returns bool array 'is_correct'.

    A correspondence is 'correct' if it lies within thr_px (Sampson distance) of the epipolar line
    implied by the ground-truth poses. (A necessary - not sufficient - condition for correctness.)
    """
    gt = rec.ds.gt_poses
    if not gt or rec.names[i] not in gt or rec.names[j] not in gt:
        return None
    Ri, ti = gt[rec.names[i]]
    Rj, tj = gt[rec.names[j]]
    Rij = Rj @ Ri.T
    tij = tj - Rij @ ti
    Kinv = np.linalg.inv(rec.K)
    Fm = Kinv.T @ (G.skew(tij) @ Rij) @ Kinv
    err = G.sampson_error(Fm, rec.kps[i][i1], rec.kps[j][i2])
    return np.sqrt(err) < thr_px
