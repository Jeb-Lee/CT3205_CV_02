"""Dense stereo reconstruction on top of the sparse SfM cameras (advanced work).

For every neighbouring pair of registered cameras (i, j):
  1. rectify the pair with the *estimated* poses (cv2.stereoRectify) so epipolar lines become image rows
  2. semi-global block matching (SGBM) -> disparity map, restricted to the depth range seen by the sparse points
  3. disparity -> 3D with the reprojection matrix Q, then rectified frame -> camera i -> world
Depth maps of different pairs are fused by voxel voting: a voxel is kept only if >= `min_support`
independent pairs put a point in it (multi-view geometric consistency), then averaged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import datasets as D
from . import geometry as G


@dataclass
class DenseConfig:
    block: int = 5
    uniqueness: int = 10
    speckle_window: int = 120
    speckle_range: int = 2
    disp12: int = 1
    pair_gap: int = 1            # partner = image k+pair_gap in the (name-sorted) registered sequence
    pixel_step: int = 2          # keep every n-th pixel of each depth map
    voxel_frac: float = 1 / 400  # voxel size as a fraction of scene radius
    min_support: int = 2         # min number of distinct pairs per voxel
    max_disp: int = 320
    depth_margin: float = 0.25


def rectify_pair(K, Ri, ti, Rj, tj, size):
    """Bouguet rectification. Returns R1, R2, P1, P2, Q (camera i is 'left', j 'right')."""
    Rrel = Rj @ Ri.T
    T = (tj - Rrel @ ti).reshape(3, 1)
    d0 = np.zeros(5)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, d0, K, d0, size, Rrel, T,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return R1, R2, P1, P2, Q


def dense_pair(rec, i, j, img_i, img_j, cfg: DenseConfig):
    """Depth map of image i from the stereo pair (i, j). Returns dict or None if the pair is unusable."""
    K = rec.K
    h, w = img_i.shape[:2]
    Ri, ti = rec.cams[i]
    Rj, tj = rec.cams[j]
    R1, R2, P1, P2, Q = rectify_pair(K, Ri, ti, Rj, tj, (w, h))
    if abs(P2[1, 3]) > 1e-6:      # vertical stereo: SGBM is horizontal-only
        return None
    if P2[0, 3] > 0:              # partner lies to the LEFT: swap roles so disparities stay positive
        return dense_pair_swapped(rec, i, j, img_i, img_j, cfg)
    f = P1[0, 0]
    B = -P2[0, 3] / f
    m1 = cv2.initUndistortRectifyMap(K, np.zeros(5), R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K, np.zeros(5), R2, P2, (w, h), cv2.CV_32FC1)
    L = cv2.remap(img_i, *m1, cv2.INTER_LINEAR)
    Rr = cv2.remap(img_j, *m2, cv2.INTER_LINEAR)

    # depth range from the sparse model (points seen in image i)
    zs = [G.depth(Ri, ti, rec.points[p][None])[0] for p, tr in rec.tracks.items() if i in tr]
    if len(zs) < 20:
        return None
    zlo, zhi = np.percentile(zs, [2, 98])
    zlo, zhi = zlo * (1 - cfg.depth_margin), zhi * (1 + cfg.depth_margin)
    dmin, dmax = f * B / zhi, f * B / zlo
    min_disp = int(max(0, np.floor(dmin)))
    if min_disp > 0.6 * w or not np.isfinite(dmin + dmax):   # rectified images barely overlap: pair unusable
        return None
    num = int(np.ceil((dmax - min_disp) / 16.0) * 16)
    num = max(16, min(cfg.max_disp, num))

    bs = cfg.block
    sgbm = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num, blockSize=bs,
                                 P1=8 * 3 * bs * bs, P2=32 * 3 * bs * bs, disp12MaxDiff=cfg.disp12,
                                 uniquenessRatio=cfg.uniqueness, speckleWindowSize=cfg.speckle_window,
                                 speckleRange=cfg.speckle_range, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(L, Rr).astype(np.float32) / 16.0
    valid = disp > (min_disp + 0.5)
    step = cfg.pixel_step
    sub = np.zeros_like(valid)
    sub[::step, ::step] = True
    valid &= sub
    xyz_rect = cv2.reprojectImageTo3D(disp, Q)          # rectified frame of camera i, units of |T|
    Xr = xyz_rect[valid]
    z = Xr[:, 2]
    ok = (z > zlo) & (z < zhi) & np.isfinite(Xr).all(axis=1)
    Xr = Xr[ok]
    col = L[valid][ok][:, ::-1]
    Xc = Xr @ R1                                         # R1^T applied to row vectors: rectified -> camera i
    Xw = (Xc - ti) @ Ri                                  # camera i -> world:  X = R^T (Xc - t)
    return dict(points=Xw, colors=col, disp=disp, left=L, right=Rr, min_disp=min_disp, num_disp=num,
                valid_frac=float(valid.sum() / disp.size), baseline=B, zrange=(float(zlo), float(zhi)))


def dense_pair_swapped(rec, i, j, img_i, img_j, cfg):
    """Partner is on the left: run the pair as (j, i) and report the depth map of j; caller only needs the 3D points."""
    return dense_pair_core(rec, j, i, img_j, img_i, cfg)


def dense_pair_core(rec, i, j, img_i, img_j, cfg):
    # identical to dense_pair but without the swap branch (avoids infinite recursion if both orders look 'left')
    K = rec.K
    h, w = img_i.shape[:2]
    Ri, ti = rec.cams[i]
    Rj, tj = rec.cams[j]
    R1, R2, P1, P2, Q = rectify_pair(K, Ri, ti, Rj, tj, (w, h))
    if abs(P2[1, 3]) > 1e-6 or P2[0, 3] > 0:
        return None
    return dense_pair(rec, i, j, img_i, img_j, cfg)


def fuse(clouds, voxel, min_support):
    """Voxel-vote fusion. clouds: list of (points, colors). Returns fused points/colors and per-point support."""
    pts = np.vstack([c[0] for c in clouds])
    cols = np.vstack([c[1] for c in clouds]).astype(np.float64)
    pid = np.concatenate([np.full(len(c[0]), k) for k, c in enumerate(clouds)])
    lo = pts.min(axis=0)
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    dims = idx.max(axis=0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    # distinct-pair support per voxel: unique (voxel, pair) combinations
    vp = np.unique(inv.astype(np.int64) * (pid.max() + 1) + pid)
    support = np.bincount(vp // (pid.max() + 1), minlength=len(uniq))
    keep_vox = support >= min_support
    sel = keep_vox[inv]
    cnt = np.bincount(inv[sel], minlength=len(uniq)).astype(np.float64)
    out_p = np.zeros((len(uniq), 3)); out_c = np.zeros((len(uniq), 3))
    for d in range(3):
        out_p[:, d] = np.bincount(inv[sel], weights=pts[sel, d], minlength=len(uniq))
        out_c[:, d] = np.bincount(inv[sel], weights=cols[sel, d], minlength=len(uniq))
    m = keep_vox
    return out_p[m] / cnt[m, None], np.clip(out_c[m] / cnt[m, None], 0, 255).astype(np.uint8), support[m]


def dense_reconstruct(rec, cfg: DenseConfig | None = None, verbose=True):
    """Run dense stereo over the registered cameras of `rec` (a finished Reconstruction)."""
    from .viz import clean_mask, robust_bounds
    cfg = cfg or DenseConfig()
    t0 = time.time()
    order = sorted(rec.cams)
    imgs = {}

    def get(k):
        if k not in imgs:
            imgs[k] = D.load_image(rec.ds, rec.names[k], rec.cfg.max_dim)[0]
        return imgs[k]

    clouds, per_pair, keep_depth = [], [], []
    for a in range(len(order) - cfg.pair_gap):
        i, j = order[a], order[a + cfg.pair_gap]
        ts = time.time()
        r = dense_pair(rec, i, j, get(i), get(j), cfg)
        if r is None:
            if verbose:
                print(f"  pair ({rec.names[i]},{rec.names[j]}): skipped")
            continue
        clouds.append((r["points"], r["colors"]))
        keep_depth.append((i, j, r))
        per_pair.append(dict(pair=[rec.names[i], rec.names[j]], points=int(len(r["points"])),
                             valid_frac=r["valid_frac"], num_disp=r["num_disp"], seconds=time.time() - ts))
        if verbose:
            print(f"  pair ({rec.names[i]},{rec.names[j]}): {len(r['points'])} pts, valid {r['valid_frac']:.2f}, "
                  f"disp {r['min_disp']}+{r['num_disp']}  ({time.time() - ts:.1f}s)", flush=True)
        for k in list(imgs):                      # keep memory bounded
            if k < i:
                del imgs[k]
    allp = np.vstack([c[0] for c in clouds])
    m = clean_mask(allp, k=5.0)
    clouds = [(c[0][clean_mask_like(c[0], allp, 5.0)], c[1][clean_mask_like(c[0], allp, 5.0)]) for c in clouds]
    Cs = np.array([G.camera_center(*rec.cams[i]) for i in order])
    c, r = robust_bounds(np.vstack([np.vstack([c[0] for c in clouds])[::50], Cs]), lo=1, hi=99)
    voxel = r * cfg.voxel_frac
    pts, cols, support = fuse(clouds, voxel, cfg.min_support)
    info = dict(n_pairs=len(clouds), raw_points=int(sum(len(c[0]) for c in clouds)), fused_points=int(len(pts)),
                voxel=float(voxel), min_support=cfg.min_support, seconds=time.time() - t0, per_pair=per_pair,
                mean_support=float(support.mean()) if len(support) else 0.0)
    if verbose:
        print(f"[dense] {info['raw_points']} raw -> {info['fused_points']} fused points "
              f"(min_support={cfg.min_support}) in {info['seconds']:.0f}s")
    return dict(xyz=pts, rgb=cols, support=support, info=info, pair_results=keep_depth)


def clean_mask_like(pts, ref, k):
    """Outlier mask for `pts` using the median / spread of the reference cloud `ref`."""
    med = np.median(ref, axis=0)
    d_ref = np.linalg.norm(ref - med, axis=1)
    return np.linalg.norm(pts - med, axis=1) < k * np.median(d_ref)
