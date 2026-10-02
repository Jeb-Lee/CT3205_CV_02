"""Report figures for the own captures: input photo next to the dense reconstruction seen from the SAME direction.

    python scripts/render_own_figures.py
Writes docs/report_figs/fig10_own_positive.png (yellow sign, fire extinguisher, snacks) and fig11_own_negative.png (bottle).
"""
import os, pickle, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import geometry as G, viz, datasets as D

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 6.5, "axes.titlesize": 6.5})
OUT = os.path.join("docs", "report_figs")


def read_ply(path):
    raw = open(path, "rb").read()
    k = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(raw[:k].split(b"element vertex ")[1].split()[0])
    r = np.frombuffer(raw[k:k + 15 * n], dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    return r["p"].astype(np.float64), r["c"].copy()


def view_from_camera(rec, cam_idx):
    """Orthonormal viewer basis (right, depth, up) looking along camera `cam_idx` (projected horizontal), up = mean camera up."""
    M = viz.level_matrix(rec)
    up = M[2]
    R, t = rec.cams[cam_idx]
    f = R.T @ np.array([0, 0, 1.0]); f -= (f @ up) * up; f /= np.linalg.norm(f)
    right = np.cross(f, up); right /= np.linalg.norm(right)
    return right, f, up


def panel(ax, rec, ply, cam_idx, maxpts=160000, s=0.5):
    P, C = read_ply(ply)
    right, f, up = view_from_camera(rec, cam_idx)
    keep = viz.clean_mask(P, 4.0)
    P, C = P[keep], C[keep]
    if len(P) > maxpts:
        sel = np.random.default_rng(0).choice(len(P), maxpts, replace=False); P, C = P[sel], C[sel]
    x, y, z = P @ right, P @ f, P @ up
    order = np.argsort(-y)                                   # far first, near last: simple occlusion
    ax.scatter(x[order], z[order], c=C[order] / 255.0, s=s, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    lo, hi = np.percentile(x, [0.5, 99.5]), np.percentile(z, [0.5, 99.5])
    pad = 0.05 * max(hi[1] - hi[0], lo[1] - lo[0])
    ax.set_xlim(lo[0] - pad, lo[1] + pad); ax.set_ylim(hi[0] - pad, hi[1] + pad)


def photo(ax, rec, ds, cam_idx):
    img, _ = D.load_image(ds, rec.names[cam_idx], 900)
    ax.imshow(img[:, :, ::-1]); ax.axis("off")


sets = [("yellow_all", "yellow_sign", "dense_object.ply", "(a) yellow sign", 0.3),
        ("fire", "Fire est", "dense_object_v2.ply", "(b) fire extinguisher", 0.3),
        ("snacks", "Snacks", "dense_object.ply", "(c) snacks", 0.3)]
fig, axs = plt.subplots(3, 2, figsize=(3.45, 4.3), gridspec_kw=dict(width_ratios=[1, 1], hspace=0.18, wspace=0.03))
for r, (tag, dname, ply, title, s) in enumerate(sets):
    rec = pickle.load(open(f"results/own/{tag}/rec.pkl", "rb"))
    ds = D.load_dataset(dname)
    cam = sorted(rec.cams)[len(rec.cams) // 2]
    photo(axs[r, 0], rec, ds, cam); panel(axs[r, 1], rec, f"results/own/{tag}/{ply}", cam, s=1.0)
    axs[r, 0].set_title(title, fontsize=5.5, loc="left", pad=2)
axs[0, 1].set_title("dense reconstruction, same view", fontsize=5.5, pad=2)
fig.savefig(os.path.join(OUT, "fig10_own_positive.png"), dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
print("wrote fig10", flush=True)

fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.9), gridspec_kw=dict(wspace=0.04))
rec = pickle.load(open("results/own/bottle_classic/rec.pkl", "rb")); ds = D.load_dataset("Bottle")
cam = sorted(rec.cams)[len(rec.cams) // 3]
photo(axs[0], rec, ds, cam); axs[0].set_title("(a) input photo", fontsize=6)
panel(axs[1], rec, "results/own/bottle_classic/dense_object.ply", cam, s=1.2); axs[1].set_title("(b) classical", fontsize=6)
rec2 = pickle.load(open("results/own/bottle_learned/rec.pkl", "rb"))
panel(axs[2], rec2, "results/own/bottle_learned/dense_object.ply", cam, s=1.2); axs[2].set_title("(c) DISK + LightGlue", fontsize=6)
fig.savefig(os.path.join(OUT, "fig11_own_negative.png"), dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
print("wrote fig11", flush=True)
