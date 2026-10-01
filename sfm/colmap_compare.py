"""Apples-to-apples comparison of our reconstruction with the COLMAP model shipped with the dataset.

The shipped COLMAP model was built from ALL images of the dataset (SIFT-GPU features, exhaustive/retrieval matching,
incremental SfM with local+global bundle adjustment and intrinsics refinement). To compare fairly on a subset of N images
we restrict COLMAP's model to the 3D points that are observed by >= 2 of OUR images and count only observations in those
images. Reprojection errors are expressed in ORIGINAL-resolution pixels for both systems (ours is scaled up from the
1600 px working resolution).
"""
from __future__ import annotations

import numpy as np

from . import evaluate as Ev
from . import geometry as G


def colmap_subset(ds, names):
    """Statistics of the COLMAP model restricted to the given images."""
    ids = np.array([ds.gt_ids[n] for n in names if n in ds.gt_ids])
    k = np.array([int(np.isin(t, ids).sum()) for t in ds.gt_tracks])
    keep = k >= 2
    return dict(
        colmap_points=int(keep.sum()),
        colmap_mean_track_len=float(k[keep].mean()) if keep.any() else 0.0,
        colmap_points_ge3=int((k >= 3).sum()),
        colmap_reproj_px=float(ds.gt_err[keep].mean()) if keep.any() else float("nan"),
        colmap_points_per_image=float(k[keep].sum() / max(len(ids), 1)),
        colmap_full_points=int(len(ds.gt_points)),
        colmap_full_mean_track_len=float(np.mean([len(t) for t in ds.gt_tracks])),
        keep_mask=keep,
    )


def compare(rec, pc=None, include_mask=False):
    """Our model vs COLMAP on the same images. Returns a flat dict."""
    ds = rec.ds
    pc = pc or rec.point_cloud(with_colors=False)
    gt = Ev.evaluate_against_gt(rec, pc)
    sim3 = gt.pop("_sim3", None)
    cs = colmap_subset(ds, [rec.names[i] for i in rec.cams])
    keep = cs.pop("keep_mask")
    scale = ds.size[0] / (rec.K[0, 2] * 2)                       # working-resolution px -> original px
    oe = rec.observation_errors()
    n_reg = max(len(rec.cams), 1)
    out = dict(
        ours_points=int(len(pc["xyz"])),
        ours_mean_track_len=float(pc["track_len"].mean()) if len(pc["xyz"]) else 0.0,
        ours_points_ge3=int((pc["track_len"] >= 3).sum()),
        ours_reproj_px=float(oe.mean() * scale) if len(oe) else float("nan"),
        ours_points_per_image=float(pc["track_len"].sum() / n_reg),
        point_ratio_ours_over_colmap=float(len(pc["xyz"]) / max(cs["colmap_points"], 1)),
        n_images=len(rec.cams),
        cam_pos_rmse_rel=gt.get("cam_pos_rmse_rel", float("nan")),
        cam_rot_mean_deg=gt.get("cam_rot_mean_deg", float("nan")),
        pt_within_2pct=gt.get("pt_within_2pct", float("nan")),
        gt_completeness_2pct=gt.get("gt_completeness_2pct", float("nan")),
        **cs,
    )
    if include_mask:
        out["_keep"], out["_sim3"] = keep, sim3
    return out
