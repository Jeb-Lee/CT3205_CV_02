"""Reconstruct one of our own captured sets (no ground truth) and log everything.

    python scripts/run_own.py --set yellow_sign --tag yellow_all --dense --video
    python scripts/run_own.py --set Bottle --tag bottle_classic
    python scripts/run_own.py --set Bottle --tag bottle_learned --detector disk --matcher lightglue --min_pair 150

Steps
  1. focal-length sweep (the phone is uncalibrated; EXIF gives f only to a few %): reconstruct 10 images at several focal
     factors and keep the one with the lowest reprojection error of points seen in >= 3 views  -> focal_sweep.json/png
  2. sparse SfM on ALL images, all-pairs matching (GPU brute force = identical to CPU, 10x faster; all pairs also link the
     first and last frame of a full orbit)
  3. statistics, per-image reprojection error (flags images whose field of view / resolution do not fit), cloud, figures
  4. optional dense stereo and turntable video
No ground truth exists for these sets: quality is judged by reprojection error, track statistics and inspection.
"""
import argparse
import json
import os
import pickle
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from sfm import datasets as D, viz, geometry as G
from sfm.reconstruction import Reconstruction, SfMConfig
from sfm.dense import DenseConfig, dense_reconstruct
from run_experiments import median_parallax

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, help="folder name under Datasets/")
ap.add_argument("--tag", required=True)
ap.add_argument("--detector", default="rootsift")
ap.add_argument("--matcher", default="gpu", help="gpu | bf | lightglue")
ap.add_argument("--min_pair", type=int, default=None, help="sets min_pair_inliers and min_pair_use")
ap.add_argument("--ba_every", type=int, default=0, help="0 = automatic (every 1 for <=30 images, else 6)")
ap.add_argument("--max_dim", type=int, default=1600)
ap.add_argument("--nfeatures", type=int, default=8000)
ap.add_argument("--f_factor", type=float, default=None, help="skip the sweep and use this focal factor")
ap.add_argument("--sweep_only", action="store_true", help="stop after the focal sweep")
ap.add_argument("--dense", action="store_true")
ap.add_argument("--video", action="store_true")
a = ap.parse_args()

OUT = os.path.join("results", "own", a.tag)
os.makedirs(OUT, exist_ok=True)
ds = D.load_dataset(a.set)
names = ds.names
n = len(names)
f0 = float(ds.K[0, 0])
print(f"=== {a.tag}: {n} images from Datasets/{a.set}; EXIF focal {f0:.1f} px at {ds.size}", flush=True)


def make_cfg(ba_every=1, **kw):
    cfg = SfMConfig(detector=a.detector, matcher=a.matcher, max_dim=a.max_dim, nfeatures=a.nfeatures, window=None,
                    ba_every=ba_every, **kw)
    if a.min_pair:
        cfg.min_pair_inliers = cfg.min_pair_use = a.min_pair
    return cfg


# ---------------------------------------------------------------------------------------- 1. focal sweep
def set_focal(f):
    ds.K[0, 0] = ds.K[1, 1] = f


def trial(fac, subset):
    set_focal(f0 * fac)
    rec = Reconstruction(ds, subset, make_cfg(ba_every=1), cache_dir="results/cache", verbose=False).run()
    st = rec.stats()
    return dict(factor=fac, f_px=f0 * fac, registered=st["n_registered"], n=len(subset), points=st["n_points"],
                reproj_mean_track3=st["reproj_mean_track3"], reproj_median=st["reproj_median"], reproj_mean=st["reproj_mean"])


if a.f_factor is None:
    start = n // 3 if n > 45 else 0
    subset = names[start:start + 24:3] if n > 30 else names[start:start + 8]
    print(f"focal sweep on {len(subset)} images ({subset[0]} .. {subset[-1]})", flush=True)
    rows = []
    for fac in (0.92, 1.00, 1.08):
        rows.append(trial(fac, subset)); print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    ok = [r for r in rows if r["registered"] >= 0.9 * r["n"]] or rows
    best = min(ok, key=lambda r: r["reproj_mean_track3"])["factor"]
    for fac in (best * 0.98, best * 1.02):
        rows.append(trial(round(fac, 4), subset)); print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    ok = [r for r in rows if r["registered"] >= 0.9 * r["n"]] or rows
    best_row = min(ok, key=lambda r: r["reproj_mean_track3"])
    fac_best = best_row["factor"]
    json.dump(dict(rows=rows, best_factor=fac_best, f_exif_px=f0, f_best_px=f0 * fac_best), open(os.path.join(OUT, "focal_sweep.json"), "w"), indent=1)
    d = pd.DataFrame(rows).sort_values("factor")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(d.factor, d.reproj_mean_track3, "o-", label=">=3-view mean"); ax.plot(d.factor, d.reproj_median, "s--", label="median (all obs)")
    ax.axvline(fac_best, color="r", ls=":"); ax.set_xlabel("focal factor (1.0 = EXIF 35mm-equivalent value)"); ax.set_ylabel("reprojection error (px)")
    ax.legend(); ax.set_title(f"{a.tag}: focal sweep, best x{fac_best:.3f}", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(OUT, "focal_sweep.png"), dpi=130); plt.close(fig)
else:
    fac_best = a.f_factor
if a.sweep_only and a.f_factor is None:
    print(f"sweep only: best factor {fac_best:.4f} -> f = {f0 * fac_best:.1f} px"); sys.exit(0)
set_focal(f0 * fac_best)
print(f"focal factor {fac_best:.3f} -> f = {f0 * fac_best:.1f} px (original resolution)", flush=True)

# ---------------------------------------------------------------------------------------- 2. full reconstruction
ba_every = a.ba_every or (1 if n <= 30 else 6)
t0 = time.time()
rec = Reconstruction(ds, names, make_cfg(ba_every=ba_every), cache_dir="results/cache", verbose=True).run()
secs = time.time() - t0
pc = rec.point_cloud()
st = rec.stats()
st["parallax_median_deg"] = median_parallax(rec, max_pts=3000)
st.update(set=a.set, tag=a.tag, focal_factor=fac_best, f_px_original=f0 * fac_best, f_exif_px=f0, ba_every=ba_every,
          detector=a.detector, matcher=a.matcher, seconds_total=secs)
unreg = [names[i] for i in range(n) if i not in rec.cams]
st["unregistered_images"] = unreg
print(f"\n[{a.tag}] registered {st['n_registered']}/{n}, {st['n_points']} points, mean track {st['mean_track_len']:.2f}, "
      f"reproj {st['reproj_mean']:.3f}px (>=3-view {st['reproj_mean_track3']:.3f}px), {secs:.0f}s", flush=True)

# per-image reprojection error + stored size
rows = []
for i in sorted(rec.cams):
    R, t = rec.cams[i]
    errs = []
    for pid, tr in rec.tracks.items():
        if i in tr:
            errs.append(G.reproj_error(rec.K, R, t, rec.points[pid][None], rec.kps[i][tr[i]][None])[0][0])
    sz = Image.open(ds.path(names[i])).size
    rows.append(dict(idx=i, image=names[i], stored_size=f"{sz[0]}x{sz[1]}", n_obs=len(errs), mean_reproj=float(np.mean(errs)) if errs else np.nan))
pi = pd.DataFrame(rows)
pi.to_csv(os.path.join(OUT, "per_image.csv"), index=False)
if pi.stored_size.nunique() > 1:
    g = pi.groupby("stored_size").agg(images=("idx", "size"), mean_reproj=("mean_reproj", "mean"), mean_obs=("n_obs", "mean"))
    print("mixed resolutions -> per-size reprojection error:\n", g.round(3).to_string(), flush=True)
    st["per_size"] = g.round(4).reset_index().to_dict("records")

# ---------------------------------------------------------------------------------------- 3. outputs
json.dump(st, open(os.path.join(OUT, "stats.json"), "w"), indent=1, default=float)
viz.write_ply(os.path.join(OUT, "sparse.ply"), pc["xyz"], pc["rgb"])
pickle.dump(rec, open(os.path.join(OUT, "rec.pkl"), "wb"))
if len(rec.cams) >= 3:
    viz.plot_scene(rec, pc, os.path.join(OUT, "scene.png"), views=((25, -60), (90, -90)),
                   title=f"{a.tag}: {len(pc['xyz']):,} points, {len(rec.cams)}/{n} cameras")
    viz.error_histogram(rec, os.path.join(OUT, "reproj_hist.png"))
if a.video and len(rec.cams) >= 6:
    viz.turntable_video(rec, pc, os.path.join(OUT, "turntable.mp4"), frames=150, size=(1280, 720), fps=30, elev=14, point_size=2.5)
if a.dense and len(rec.cams) >= 6:
    t1 = time.time()
    dense = dense_reconstruct(rec, DenseConfig(), verbose=True)
    viz.write_ply(os.path.join(OUT, "dense.ply"), dense["xyz"], dense["rgb"])
    st["dense_points"] = int(len(dense["xyz"])); st["dense_seconds"] = time.time() - t1; st["dense_info"] = dense["info"]
    json.dump(st, open(os.path.join(OUT, "stats.json"), "w"), indent=1, default=float)
print("saved to", OUT, flush=True)
