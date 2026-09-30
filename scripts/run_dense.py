"""Dense stereo reconstruction (advanced work): sparse SfM cameras -> rectified SGBM depth maps -> fused dense cloud.

    python scripts/run_dense.py --start 40 --count 16 --stride 1 --out results/dense
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, viz, evaluate as Ev, geometry as G
from sfm.dense import DenseConfig, dense_reconstruct
from sfm.reconstruction import Reconstruction, SfMConfig


def plane_fit(xyz, thr, iters=800, seed=0):
    """RANSAC dominant plane: returns (n_inliers, rms residual of inliers)."""
    rng = np.random.default_rng(seed)
    best = (0, None)
    n = len(xyz)
    for _ in range(iters):
        p = xyz[rng.choice(n, 3, replace=False)]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm /= ln
        d = np.abs((xyz - p[0]) @ nrm)
        c = int((d < thr).sum())
        if c > best[0]:
            best = (c, (p[0], nrm))
    p0, nrm = best[1]
    inl = np.abs((xyz - p0) @ nrm) < thr
    # least-squares refit on inliers
    P = xyz[inl]
    c = P.mean(0)
    nrm = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    res = (P - c) @ nrm
    return int(inl.sum()), float(np.sqrt(np.mean(res ** 2)))


def pair_figure(r, path, name_i, name_j):
    L, R, disp = r["left"], r["right"], r["disp"]
    h, w = L.shape[:2]
    stack = np.vstack([L, R]).copy()
    for y in range(40, 2 * h, 70):
        cv2.line(stack, (0, y), (w - 1, y), (0, 255, 255), 1)
    dv = np.clip((disp - r["min_disp"]) / max(r["num_disp"], 1), 0, 1)
    dc = cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dc[disp <= r["min_disp"] + 0.5] = 0
    left_lbl = cv2.putText(L.copy(), f"rectified {name_i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    disp_lbl = cv2.putText(dc, "SGBM disparity (black = rejected)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    top = np.hstack([left_lbl, disp_lbl])
    epi = cv2.resize(stack, (top.shape[1], int(stack.shape[0] * top.shape[1] / stack.shape[1])))
    cv2.imwrite(path, np.vstack([top, epi]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def compare_render(sparse_pc, dense, cams, path):
    fig = plt.figure(figsize=(20, 8))
    for k, (title, xyz, rgb, s) in enumerate([("sparse SfM", sparse_pc["xyz"], sparse_pc["rgb"], 1.5),
                                              ("dense stereo (fused)", dense["xyz"], dense["rgb"], 0.25)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        m = viz.clean_mask(xyz, 3.0)
        pts, col = xyz[m], rgb[m] / 255.0
        if len(pts) > 250000:
            sel = np.random.default_rng(0).choice(len(pts), 250000, replace=False)
            pts, col = pts[sel], col[sel]
        c, r = viz.robust_bounds(np.vstack([xyz[m], cams]), lo=0, hi=100, pad=0.05)
        ax.scatter(*pts.T, c=col, s=s, depthshade=False, linewidths=0)
        ax.plot(*cams.T, "r.-", lw=0.7)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=-70, azim=-90); ax.set_axis_off()
        ax.set_title(f"{title}: {len(xyz):,} points", fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def coverage(gt_pts, cloud_gt_frame, extent, tol=0.02):
    d, _ = cKDTree(cloud_gt_frame).query(gt_pts)
    return dict(median=float(np.median(d) / extent), within=float(np.mean(d < tol * extent)))


def visible_gt_points(ds, rec, order):
    """COLMAP points that fall inside at least one of OUR cameras (ground-truth poses). Occlusion is ignored.

    Coverage must be measured only over this subset: COLMAP used all 128 images of the whole scene.
    """
    w, h = rec.K[0, 2] * 2, rec.K[1, 2] * 2
    vis = np.zeros(len(ds.gt_points), bool)
    for i in order:
        R, t = ds.gt_poses[rec.names[i]]
        uv, z = G.project(rec.K, R, t, ds.gt_points)
        vis |= (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return ds.gt_points[vis]


def analyse(ds, rec, sparse_pc, dense, sim3, extent, ext_est, cams, gt_vis):
    s, R, t = sim3
    dense_gt = s * dense["xyz"] @ R.T + t
    sparse_gt = s * sparse_pc["xyz"] @ R.T + t
    tree_gt = cKDTree(ds.gt_points)
    # scene radius of the reconstructed volume (GT frame): the camera arc is much smaller than the scene, so
    # normalising by trajectory extent overstates errors; use the cloud's own robust radius, and only COLMAP
    # points inside that volume for the coverage test
    dm = viz.clean_mask(dense_gt, 3.0)
    c_s, r_s = viz.robust_bounds(dense_gt[dm], lo=1, hi=99, pad=0.0)
    in_vol = np.all(np.abs(gt_vis - c_s) < r_s, axis=1)
    gt_in = gt_vis[in_vol]
    res = dict(
        sparse_points=int(len(sparse_pc["xyz"])), dense_points=int(len(dense["xyz"])),
        density_gain=float(len(dense["xyz"]) / max(len(sparse_pc["xyz"]), 1)),
        n_colmap_visible=int(len(gt_vis)),
        n_colmap_in_volume=int(len(gt_in)), scene_radius_gt=float(r_s),
        colmap_coverage_sparse=coverage(gt_vis, sparse_gt, extent),
        colmap_coverage_dense=coverage(gt_vis, dense_gt, extent),
        colmap_coverage_sparse_scene=coverage(gt_in, sparse_gt, r_s),
        colmap_coverage_dense_scene=coverage(gt_in, dense_gt, r_s),
        dense_to_colmap_median_scene=float(np.median(tree_gt.query(dense_gt[dm])[0]) / r_s),
        sparse_to_colmap_median_scene=float(np.median(tree_gt.query(sparse_gt)[0]) / r_s),
        dense_to_colmap_median=float(np.median(tree_gt.query(dense_gt)[0]) / extent),
        sparse_to_colmap_median=float(np.median(tree_gt.query(sparse_gt)[0]) / extent),
    )
    m_d = viz.clean_mask(dense["xyz"], 3.0)
    m_s = viz.clean_mask(sparse_pc["xyz"], 3.0)
    dp = dense["xyz"][m_d]
    dp = dp[np.random.default_rng(0).choice(len(dp), min(len(dp), 120000), replace=False)]
    for nm, pts in (("sparse", sparse_pc["xyz"][m_s]), ("dense", dp)):
        n_in, rms = plane_fit(pts, 0.01 * ext_est)
        res[f"plane_{nm}"] = dict(inliers=n_in, inlier_frac=n_in / len(pts), rms_rel_extent=rms / ext_est)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="south-building")
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gap", type=int, default=1, help="stereo partner = k+gap")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--support", type=int, default=2)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--video", action="store_true", help="render dense turntable mp4")
    ap.add_argument("--out", default="results/dense")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds = D.load_dataset(a.dataset)
    names = D.select_images(ds, a.start, a.count, a.stride)
    import pickle
    pk = os.path.join("results", "cache", f"rec_{a.dataset}_{a.start}_{a.count}_{a.stride}.pkl")
    if os.path.isfile(pk):
        rec = pickle.load(open(pk, "rb"))
        print("loaded cached SfM reconstruction", pk)
    else:
        rec = Reconstruction(ds, names, SfMConfig(), cache_dir="results/cache").run()
        pickle.dump(rec, open(pk, "wb"))
    sparse_pc = rec.point_cloud()
    gt = Ev.evaluate_against_gt(rec, sparse_pc)
    s, R, t = gt.pop("_sim3")
    print("sparse vs GT:", {k: round(v, 4) for k, v in gt.items() if isinstance(v, float)})

    cfg = DenseConfig(block=a.block, pair_gap=a.gap, min_support=a.support)
    dense = dense_reconstruct(rec, cfg)
    info = dense["info"]

    # ------------------------------------------------------------- outputs
    viz.write_ply(os.path.join(a.out, "dense.ply"), dense["xyz"], dense["rgb"])
    viz.write_ply(os.path.join(a.out, "sparse.ply"), sparse_pc["xyz"], sparse_pc["rgb"])
    order = sorted(rec.cams)
    cams = np.array([G.camera_center(*rec.cams[i]) for i in order])
    viz.write_cameras_ply(os.path.join(a.out, "cameras.ply"), rec, scale=0.15 * viz.robust_bounds(dense["xyz"])[1])
    compare_render(sparse_pc, dense, cams, os.path.join(a.out, "sparse_vs_dense.png"))
    i, j, r = max(dense["pair_results"], key=lambda x: x[2]["valid_frac"])
    pair_figure(r, os.path.join(a.out, "stereo_pair.jpg"), rec.names[i], rec.names[j])

    # ------------------------------------------------------------- quantitative comparison
    C_gt = np.array([G.camera_center(*ds.gt_poses[rec.names[i]]) for i in order])
    extent = float(np.linalg.norm(C_gt - C_gt.mean(0), axis=1).max())
    ext_est = float(np.linalg.norm(cams - cams.mean(0), axis=1).max())
    gt_vis = visible_gt_points(ds, rec, order)
    res = analyse(ds, rec, sparse_pc, dense, (s, R, t), extent, ext_est, cams, gt_vis)
    print(json.dumps(res, indent=1))
    res.update(sparse_vs_gt={k: v for k, v in gt.items() if isinstance(v, float)}, dense_info={k: v for k, v in info.items()},
               config=cfg.__dict__, images=names)

    if a.ablate:   # one SfM run, several dense configurations
        import pandas as pd
        rows = []
        for label, kw in [("block=3", dict(block=3)), ("block=5 (default)", dict()), ("block=9", dict(block=9)),
                          ("gap=2", dict(pair_gap=2)), ("gap=3", dict(pair_gap=3)),
                          ("support=1", dict(min_support=1)), ("support=3", dict(min_support=3))]:
            c = DenseConfig(**{**dict(block=a.block, pair_gap=a.gap, min_support=a.support), **kw})
            d = dense_reconstruct(rec, c, verbose=False)
            r = analyse(ds, rec, sparse_pc, d, (s, R, t), extent, ext_est, cams, gt_vis)
            rows.append(dict(config=label, pairs=d["info"]["n_pairs"], raw_points=d["info"]["raw_points"],
                             fused_points=r["dense_points"], seconds=d["info"]["seconds"],
                             dense_to_colmap_median=r["dense_to_colmap_median"],
                             coverage_within2pct_scene=r["colmap_coverage_dense_scene"]["within"],
                             dense_to_colmap_median_scene=r["dense_to_colmap_median_scene"],
                             plane_rms=r["plane_dense"]["rms_rel_extent"], plane_inlier_frac=r["plane_dense"]["inlier_frac"]))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "ablation.csv"), index=False)
        res["ablation"] = rows
    json.dump(res, open(os.path.join(a.out, "dense_stats.json"), "w"), indent=2, default=float)
    if a.video:
        sel = np.random.default_rng(0).choice(len(dense["xyz"]), min(len(dense["xyz"]), 150000), replace=False)
        pc = dict(xyz=dense["xyz"][sel], rgb=dense["rgb"][sel])
        viz.turntable_video(rec, pc, os.path.join(a.out, "dense_turntable.mp4"), frames=120, size=(1280, 720),
                            elev=8, point_size=0.7)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
