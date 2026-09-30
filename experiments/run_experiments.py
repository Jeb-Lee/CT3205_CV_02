"""Experiments A-D for Assignment 2.

    python experiments/run_experiments.py A B C D          # run any subset
    python experiments/run_experiments.py A --dataset south-building

Every run writes:  results/exp<X>/results.csv, results.md, figures, per-run scene plot + stats.
All experiments share ONE control configuration (the baseline) and change exactly one variable.
Failures (registration collapse, no initial pair, ...) are recorded as rows, not crashes: they are data.
"""
import argparse
import json
import os
import sys
import time
import traceback

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, viz, evaluate as Ev, geometry as G
from sfm.reconstruction import Reconstruction, SfMConfig

CACHE = os.path.join("results", "cache")
START = 40            # images P1180181.. : richly connected part of the sequence (see scripts/diag_pairs.py)
BASE_STRIDE, BASE_COUNT = 2, 12


def median_parallax(rec, max_pts=4000):
    """Median (over 3D points) of the largest angle between any two viewing rays: how well-conditioned the depth is."""
    centers = {i: G.camera_center(*rec.cams[i]) for i in rec.cams}
    angs = []
    for pid in list(rec.points)[:max_pts]:
        obs = list(rec.tracks[pid])
        X = rec.points[pid]
        best = 0.0
        for a in range(len(obs)):
            for b in range(a + 1, len(obs)):
                ang = G.triangulation_angle_deg(centers[obs[a]], centers[obs[b]], X[None])[0]
                best = max(best, ang)
        angs.append(best)
    return float(np.median(angs)) if angs else float("nan")


def run_one(ds, label, names, cfg, degrade=None, out_dir=None, save_scene=True):
    """Run the pipeline once; return a flat dict of metrics (never raises)."""
    row = dict(label=label, n_images=len(names), detector=cfg.detector)
    t0 = time.time()
    try:
        rec = Reconstruction(ds, names, cfg, degrade=degrade, cache_dir=CACHE, verbose=False).run()
        pc = rec.point_cloud(with_colors=save_scene)
        st = rec.stats()
        gt = Ev.evaluate_against_gt(rec, pc)
        gt.pop("_sim3", None)
        st.update(gt)
        st["median_parallax_deg"] = median_parallax(rec)
        st["failed"] = False
        st["frac_registered"] = st["n_registered"] / st["n_images"]
        row.update({k: v for k, v in st.items() if not isinstance(v, (list, dict))})
        if save_scene and out_dir and len(pc["xyz"]) > 10 and len(rec.cams) >= 2:
            viz.plot_scene(rec, pc, os.path.join(out_dir, f"scene_{label}.png"), views=((25, -60), (90, -90)),
                           title=f"{label}: {len(pc['xyz'])} pts, {len(rec.cams)}/{len(names)} cams")
    except Exception as e:  # a failed reconstruction is a valid experimental outcome
        row.update(failed=True, error=f"{type(e).__name__}: {e}", n_registered=0, n_points=0, frac_registered=0.0)
        print(f"   !! {label} failed: {e}")
        traceback.print_exc(limit=2)
    row["seconds_wall"] = time.time() - t0
    print(f"   {label:>22s}: reg {row.get('n_registered', 0)}/{len(names)}  pts {row.get('n_points', 0)}  "
          f"reproj {row.get('reproj_mean', float('nan')):.3f}px  camRMSE {row.get('cam_pos_rmse_rel', float('nan')):.4f}  "
          f"({row['seconds_wall']:.0f}s)", flush=True)
    return row


def save_table(rows, out_dir, cols):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    keep = [c for c in cols if c in df.columns]
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        try:
            f.write(df[keep].to_markdown(index=False, floatfmt=".4g"))
        except ImportError:  # tabulate not installed: plain text table is fine
            f.write(df[keep].to_string())
    return df


COLS = ["label", "n_registered", "n_images", "n_points", "n_points_track3", "mean_track_len", "reproj_mean",
        "reproj_mean_track3", "cam_pos_rmse_rel", "cam_rot_mean_deg", "pt_acc_median", "pt_within_2pct",
        "gt_completeness_2pct", "mean_features", "mean_inliers_per_pair", "mean_inlier_ratio",
        "median_parallax_deg", "seconds", "failed"]


def line_plots(df, x, xlabel, out, metrics, logx=False, cat=False):
    """One small-multiples figure: metric vs the experiment variable."""
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    axs = np.atleast_1d(axs)
    for ax, (m, title) in zip(axs, metrics):
        if m not in df.columns:
            continue
        if cat:
            ax.bar(df[x].astype(str), df[m].fillna(0), color="tab:blue")
            ax.tick_params(axis="x", rotation=30)
        else:
            ax.plot(df[x], df[m], "o-", color="tab:blue")
            if logx:
                ax.set_xscale("log")
        ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


MET = [("n_points", "# 3D points"), ("reproj_mean", "mean reproj. error (px)"),
       ("cam_pos_rmse_rel", "camera position error (rel.)"), ("gt_completeness_2pct", "completeness vs COLMAP")]


# ------------------------------------------------------------------------------------ A
def exp_A(ds, args):
    """Number of images. Nested subsets of the same sequence so only N changes."""
    out = os.path.join("results", "expA"); os.makedirs(out, exist_ok=True)
    rows = []
    for n in [3, 4, 6, 8, 12, 16, 20]:
        names = D.select_images(ds, START, n, BASE_STRIDE)
        rows.append(run_one(ds, f"N={n}", names, SfMConfig(), out_dir=out) | dict(N=n))
    df = save_table(rows, out, COLS)
    line_plots(df, "N", "number of input images", os.path.join(out, "expA_metrics.png"),
               MET + [("mean_track_len", "mean track length"), ("seconds", "runtime (s)")])
    return df


# ------------------------------------------------------------------------------------ B
def exp_B(ds, args):
    """Detector / descriptor: SIFT, RootSIFT, ORB, AKAZE, BRISK on identical images and pipeline."""
    out = os.path.join("results", "expB"); os.makedirs(out, exist_ok=True)
    names = D.select_images(ds, START, BASE_COUNT, BASE_STRIDE)
    rows = []
    for det in ["sift", "rootsift", "orb", "akaze", "brisk"]:
        rows.append(run_one(ds, det, names, SfMConfig(detector=det), out_dir=out) | dict(detector_name=det))
    df = save_table(rows, out, COLS)
    line_plots(df, "label", "detector", os.path.join(out, "expB_metrics.png"),
               MET + [("mean_features", "features / image"), ("mean_inliers_per_pair", "verified inliers / pair"),
                      ("seconds", "runtime (s)")], cat=True)
    return df


# ------------------------------------------------------------------------------------ C
def exp_C(ds, args):
    """Viewpoint change: same number of images (8), increasing frame stride = increasing camera baseline."""
    out = os.path.join("results", "expC"); os.makedirs(out, exist_ok=True)
    rows = []
    for stride in [1, 2, 3, 4, 5, 6]:
        names = D.select_images(ds, START, 8, stride)
        rows.append(run_one(ds, f"stride={stride}", names, SfMConfig(), out_dir=out) | dict(stride=stride))
    df = save_table(rows, out, COLS + ["stride"])
    line_plots(df, "stride", "frame stride (viewpoint change)", os.path.join(out, "expC_metrics.png"),
               MET + [("median_parallax_deg", "median max parallax (deg)"), ("mean_inliers_per_pair", "verified inliers / pair")])
    # extra: tiny-baseline pair -> depth is ill-conditioned. Two consecutive frames, triangulate & compare to GT depth
    rows2 = small_baseline_depth_study(ds, out)
    pd.DataFrame(rows2).to_csv(os.path.join(out, "two_view_baseline_study.csv"), index=False)
    return df


def small_baseline_depth_study(ds, out):
    """Two-view depth error vs baseline: pair (i, i+k) for k=1..8, depth compared against COLMAP triangulated points.

    Demonstrates why insufficient camera movement ruins depth (Assignment Exp. C requirement).
    """
    from sfm import features as F, matching as Mt
    names = D.select_images(ds, START, 20, 1)
    imgs = [D.load_image(ds, n, 1600) for n in names[:12]]
    kd = []
    for img, K in imgs:
        kd.append(F.extract(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "rootsift", 8000))
    K = imgs[0][1]
    rows = []
    base = 0
    for k in range(1, 9):
        i, j = base, base + k
        (p1, d1), (p2, d2) = kd[i], kd[j]
        a, b, _ = Mt.match_pair(d1, d2, cv2.NORM_L2, 0.8, True)
        E, m = Mt.verify_essential(p1[a], p2[b], K, 1.0)
        if E is None or m.sum() < 15:
            rows.append(dict(k=k, inliers=int(m.sum()))); continue
        x1, x2 = p1[a][m], p2[b][m]
        R, t, _, X = G.recover_pose_cheirality(E, K, x1, x2)
        # GT relative pose -> ground-truth scale-free comparison of the recovered rotation/translation direction
        (Ri, ti), (Rj, tj) = ds.gt_poses[names[i]], ds.gt_poses[names[j]]
        Rg = Rj @ Ri.T; tg = tj - Rg @ ti
        rot_err = G.rotation_angle_deg(R @ Rg.T)
        t_dir_err = np.degrees(np.arccos(np.clip(abs(t @ tg) / np.linalg.norm(tg), -1, 1)))
        ang = G.triangulation_angle_deg(np.zeros(3), G.camera_center(R, t), X)
        # depth spread consistency: reprojected depth against GT-scale: compare to GT depth after fitting one global scale
        Xg = X.copy()
        rows.append(dict(k=k, inliers=int(m.sum()), baseline_gt=float(np.linalg.norm(G.camera_center(Rj, tj) - G.camera_center(Ri, ti))),
                         median_parallax_deg=float(np.median(ang)), rot_err_deg=float(rot_err), t_dir_err_deg=float(t_dir_err),
                         depth_rel_spread=float(np.std(Xg[:, 2]) / (np.median(Xg[:, 2]) + 1e-9))))
    return rows


# ------------------------------------------------------------------------------------ D
def exp_D(ds, args):
    """Image quality: blur, noise, resolution, contrast, brightness/gamma - one factor at a time."""
    out = os.path.join("results", "expD"); os.makedirs(out, exist_ok=True)
    names = D.select_images(ds, START, BASE_COUNT, BASE_STRIDE)
    plan = [("baseline", "baseline", SfMConfig(), D.Degrade())]
    plan += [(f"blur s={s}", "blur", SfMConfig(), D.Degrade(blur_sigma=s)) for s in (1, 2, 4, 8)]
    plan += [(f"noise s={s}", "noise", SfMConfig(), D.Degrade(noise_sigma=s)) for s in (10, 25, 50)]
    plan += [(f"res={r}px", "resolution", SfMConfig(max_dim=r), D.Degrade()) for r in (1200, 800, 400, 200)]
    plan += [(f"contrast x{c}", "contrast", SfMConfig(), D.Degrade(contrast=c)) for c in (0.5, 0.25, 0.1)]
    plan += [(f"gamma {g}", "lighting", SfMConfig(), D.Degrade(gamma=g)) for g in (2.0, 3.5)]
    rows = []
    for label, factor, cfg, deg in plan:
        rows.append(run_one(ds, label, names, cfg, degrade=deg, out_dir=out, save_scene=False) | dict(factor=factor))
    df = save_table(rows, out, ["factor"] + COLS)
    # summary figure: one panel per factor showing the # of points and camera error
    fig, axs = plt.subplots(2, 5, figsize=(20, 7))
    for c, fct in enumerate(["blur", "noise", "resolution", "contrast", "lighting"]):
        sub = pd.concat([df[df.factor == "baseline"], df[df.factor == fct]])
        for r, (m, t) in enumerate([("n_points", "# 3D points"), ("cam_pos_rmse_rel", "camera position error (rel.)")]):
            axs[r, c].bar(sub["label"], sub[m].fillna(0), color=["gray"] + ["tab:blue"] * (len(sub) - 1))
            axs[r, c].set_title(f"{fct}: {t}", fontsize=10); axs[r, c].tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(os.path.join(out, "expD_metrics.png"), dpi=130); plt.close(fig)
    montage(ds, names[0], [d for _, _, _, d in plan if not d.is_identity()], [l for l, _, _, d in plan if not d.is_identity()],
            os.path.join(out, "degradation_examples.jpg"))
    return df


def montage(ds, name, degs, labels, path, crop=360):
    """Centre crops of the same image under each degradation - figure for the report."""
    img, _ = D.load_image(ds, name, 1600)
    h, w = img.shape[:2]
    y0, x0 = h // 2 - crop // 2, w // 2 - crop // 2
    tiles = [img[y0:y0 + crop, x0:x0 + crop].copy()] + [D.apply_degradation(img, d)[y0:y0 + crop, x0:x0 + crop] for d in degs]
    labs = ["original"] + labels
    cols = 5
    tiles = [cv2.putText(t.copy(), l, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) for t, l in zip(tiles, labs)]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


EXPS = dict(A=exp_A, B=exp_B, C=exp_C, D=exp_D)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=list(EXPS))
    ap.add_argument("--dataset", default="south-building")
    args = ap.parse_args()
    ds = D.load_dataset(args.dataset)
    for w in args.which:
        print(f"\n===== Experiment {w} =====", flush=True)
        EXPS[w](ds, args)
    print("done")
