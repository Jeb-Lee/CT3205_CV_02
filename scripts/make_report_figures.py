"""Column-width figures for the IEEE two-column report (8.5 cm wide), regenerated from saved results.

    python scripts/make_report_figures.py            # writes docs/report_figs/fig*.png (300 dpi)
Re-runs the baseline reconstruction once (about 1 min, features cached) so figures use the real model.
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, viz, geometry as G, matching as Mt, features as F, evaluate as Ev
from sfm.colmap_compare import compare
from sfm.reconstruction import Reconstruction, SfMConfig

OUT = os.path.join("docs", "report_figs")
os.makedirs(OUT, exist_ok=True)
W = 3.45  # inches: one IEEE column
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 6.5,
                     "axes.labelsize": 6.5, "axes.titlesize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "figure.dpi": 100, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name, flush=True)


def label_img(im, text, org=(8, 26), scale=0.8):
    im = im.copy()
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return im


# ------------------------------------------------------------------------------------------- data
south = D.load_dataset("south-building")
hall = D.load_dataset("person-hall")
names = D.select_images(south, 40, 12, 2)
print("running baseline reconstruction ...", flush=True)
rec = Reconstruction(south, names, SfMConfig(), cache_dir="results/cache", verbose=False).run()
pc = rec.point_cloud()
M = viz.level_matrix(rec)

# ------------------------------------------------------------------------------------------- Fig 1: datasets
tiles = []
for ds, nm, tag in ((south, names[0], "a"), (south, names[5], "b"), (south, names[11], "c"),
                    (hall, "IMG_1120.JPG", "d"), (hall, "IMG_1062.JPG", "e"), (hall, "IMG_1226.JPG", "f")):
    im, _ = D.load_image(ds, nm, 640)
    im = cv2.resize(im, (640, 427))
    tiles.append(label_img(im, f"({tag})", org=(12, 56), scale=2.3))
grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
cv2.imwrite(os.path.join(OUT, "fig1_datasets.png"), grid)
print("wrote fig1", flush=True)

# ------------------------------------------------------------------------------------------- Fig 2: matching cascade
i, j = rec.init_pair
im1, _ = D.load_image(south, names[i], 560)
im2, _ = D.load_image(south, names[j], 560)
sx = 560 / max(rec.K[0, 2] * 2, 1)
norm = cv2.NORM_L2
r1, r2 = Mt.raw_nn(rec.descs[i], rec.descs[j], norm)
a1, a2, _ = Mt.match_pair(rec.descs[i], rec.descs[j], norm, rec.cfg.ratio, rec.cfg.cross_check)
v1, v2 = rec.pair_matches(i, j)
rows = []
counts = {}
for tag, (x1, x2), txt in (("a", (r1, r2), "raw nearest neighbour"), ("b", (a1, a2), "ratio test + mutual check"),
                           ("c", (v1, v2), "after RANSAC on E")):
    ok = Ev.match_correctness(rec, i, j, x1, x2)
    col = [(0, 200, 0) if o else (0, 0, 255) for o in ok]
    canvas = viz.draw_matches(im1, im2, rec.kps[i][x1] * sx, rec.kps[j][x2] * sx, col, 70, thickness=1)
    nwrong = int((~ok).sum())
    counts[tag] = dict(stage=txt, matches=int(len(x1)), wrong=nwrong)
    strip = np.zeros((38, canvas.shape[1], 3), np.uint8)
    cv2.putText(strip, f"({tag}) {txt}: {len(x1)} matches, {nwrong} wrong", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    rows.append(np.vstack([strip, canvas]))
import json
json.dump(counts, open(os.path.join(OUT, "fig2_counts.json"), "w"), indent=1)
print("fig2 counts", counts, flush=True)
cv2.imwrite(os.path.join(OUT, "fig2_matching.png"), np.vstack(rows))
print("wrote fig2", flush=True)

# ------------------------------------------------------------------------------------------- Fig 3: reconstruction
xyz = pc["xyz"] @ M.T
Cs = np.array([G.camera_center(*rec.cams[k]) for k in rec.cams]) @ M.T
dirs = np.array([rec.cams[k][0].T @ np.array([0, 0, 1.0]) for k in rec.cams]) @ M.T
keep = viz.clean_mask(xyz, 2.5)
col = pc["rgb"][keep] / 255.0
fig, ax = plt.subplots(1, 2, figsize=(W, 1.75))
for a, (u, v, un, vn) in zip(ax, ((0, 1, "x", "y"), (0, 2, "x", "z (up)"))):
    a.scatter(xyz[keep][:, u], xyz[keep][:, v], c=col, s=0.15, linewidths=0, rasterized=True)
    a.quiver(Cs[:, u], Cs[:, v], dirs[:, u], dirs[:, v], color="tab:red", width=0.012, scale=9, headwidth=3.2, zorder=3)
    a.plot(Cs[:, u], Cs[:, v], "k:", lw=0.4)
    a.set_aspect("equal"); a.set_xlabel(un); a.set_ylabel(vn)
    a.tick_params(length=2)
ax[0].set_title("(a) plan view", fontsize=6.5); ax[1].set_title("(b) front view", fontsize=6.5)
fig.tight_layout(pad=0.3)
save(fig, "fig3_reconstruction.png")

# ------------------------------------------------------------------------------------------- Fig 4: reprojection
regs = sorted(rec.cams)
im_id = regs[len(regs) // 2]
img, _ = D.load_image(south, names[im_id], rec.cfg.max_dim)
R, t = rec.cams[im_id]
det, rep = [], []
for pid, tr in rec.tracks.items():
    if im_id in tr:
        x = rec.kps[im_id][tr[im_id]]
        uv, z = G.project(rec.K, R, t, rec.points[pid][None])
        det.append(x); rep.append(uv[0])
det, rep = np.array(det), np.array(rep)
cx, cy = np.median(det[:, 0]), np.median(det[:, 1])
x0, y0, ww, hh = int(cx - 70), int(cy - 48), 140, 96
sel = (det[:, 0] > x0) & (det[:, 0] < x0 + ww) & (det[:, 1] > y0) & (det[:, 1] < y0 + hh)
e_all = rec.observation_errors(2) * 1.0
e_3 = rec.observation_errors(3) * 1.0
fig, ax = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax[0].imshow(img[:, :, ::-1]); ax[0].set_xlim(x0, x0 + ww); ax[0].set_ylim(y0 + hh, y0); ax[0].axis("off")
mag = 15.0
d_, r_ = det[sel], rep[sel]
if len(d_) > 55:
    pick = np.random.default_rng(1).choice(len(d_), 55, replace=False); d_, r_ = d_[pick], r_[pick]
ax[0].scatter(*d_.T, s=22, facecolors="none", edgecolors="lime", linewidths=0.7, label="detected")
ax[0].scatter(*(d_ + mag * (r_ - d_)).T, s=7, c="red", linewidths=0, label=f"reprojected (offset x{int(mag)})")
for a_, b_ in zip(d_, d_ + mag * (r_ - d_)):
    ax[0].plot([a_[0], b_[0]], [a_[1], b_[1]], color="yellow", lw=0.4)
ax[0].legend(loc="lower right", fontsize=4.8, framealpha=0.8, handletextpad=0.2, borderpad=0.2)
ax[0].set_title("(a) detected vs reprojected", fontsize=6.5)
bins = np.linspace(0, 1.0, 41)
ax[1].hist(np.clip(e_all, 0, 1.0), bins=bins, color="tab:blue", alpha=0.75, label=f"all obs. (median {np.median(e_all):.2f})")
ax[1].hist(np.clip(e_3, 0, 1.0), bins=bins, color="tab:orange", alpha=0.65, label=f">=3-view (median {np.median(e_3):.2f})")
ax[1].set_xlabel("reprojection error (px)"); ax[1].set_ylabel("observations")
ax[1].legend(fontsize=5, loc="upper right"); ax[1].set_title("(b) error histogram", fontsize=6.5); ax[1].tick_params(length=2)
fig.tight_layout(pad=0.3)
save(fig, "fig4_reprojection.png")

# ------------------------------------------------------------------------------------------- Fig 5: Exp A and C
A = pd.read_csv("results/expA/results.csv")
C = pd.read_csv("results/expC/results.csv")
fig, ax = plt.subplots(1, 2, figsize=(W, 1.7))
a = ax[0]
a.plot(A.N, A.n_points / 1000, "o-", color="tab:blue", ms=3, lw=0.8)
a.set_xlabel("number of images N"); a.set_ylabel("3D points (thousands)", color="tab:blue"); a.tick_params(length=2)
a2 = a.twinx()
a2.plot(A.N, A.cam_pos_rmse_rel * 100, "s--", color="tab:red", ms=3, lw=0.8)
a2.set_ylabel("camera position error (%)", color="tab:red"); a2.tick_params(length=2)
for n_, nr, y_ in zip(A.N, A.n_registered, A.n_points / 1000):
    if nr != n_:
        a.annotate(f"{nr}/{n_} reg.", (n_, y_), textcoords="offset points", xytext=(2, 6), fontsize=5)
a.set_title("(a) Exp. A: number of images", fontsize=6.5)
b = ax[1]
b.bar(C.stride, C.n_registered, color="tab:green", alpha=0.7, width=0.6)
b.set_ylim(0, 9); b.set_xlabel("frame stride (8 images)"); b.set_ylabel("images registered (of 8)", color="tab:green"); b.tick_params(length=2)
b2 = b.twinx()
b2.semilogy(C.stride, C.mean_inliers_per_pair, "o-", color="tab:purple", ms=3, lw=0.8)
b2.set_ylabel("verified inliers per pair", color="tab:purple"); b2.tick_params(length=2)
b.set_title("(b) Exp. C: viewpoint change", fontsize=6.5)
fig.tight_layout(pad=0.4)
save(fig, "fig5_expA_expC.png")

# ------------------------------------------------------------------------------------------- Fig 6: Exp D
Dd = pd.read_csv("results/expD/results.csv")
base = float(Dd[Dd.factor == "baseline"].n_points.iloc[0])
Dd["pct"] = 100 * Dd.n_points / base
sub = Dd[Dd.factor != "baseline"].reset_index(drop=True)
palette = {"blur": "tab:blue", "noise": "tab:orange", "resolution": "tab:green", "contrast": "tab:purple", "lighting": "tab:gray"}
fig, a = plt.subplots(figsize=(W, 2.35))
y = np.arange(len(sub))[::-1]
a.barh(y, sub.pct, color=[palette[f] for f in sub.factor], height=0.7)
a.axvline(100, color="k", lw=0.5, ls="--")
for yy, p, nr, nm, lab in zip(y, sub.pct, sub.n_registered, sub.n_images, sub.label):
    a.text(p + 1.5, yy, f"{nr}/{nm}", va="center", fontsize=5)
a.set_yticks(y); a.set_yticklabels(sub.label, fontsize=5.5)
a.set_xlabel("3D points kept (% of undegraded baseline); annotation = images registered"); a.set_xlim(0, 135)
a.tick_params(length=2)
fig.tight_layout(pad=0.3)
save(fig, "fig6_expD.png")

# ------------------------------------------------------------------------------------------- Fig 7: dense
rec16 = pd.read_pickle("results/cache/rec_south-building_40_16_1.pkl") if False else None
import pickle
r16 = pickle.load(open("results/cache/rec_south-building_40_16_1.pkl", "rb"))
M16 = viz.level_matrix(r16)


def read_ply(path):
    raw = open(path, "rb").read()
    k = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(raw[:k].split(b"element vertex ")[1].split()[0])
    rr = np.frombuffer(raw[k:k + 15 * n], dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    return rr["p"].astype(np.float64), rr["c"].copy()


sp, sc = read_ply("results/dense/sparse.ply")
dp, dc = read_ply("results/dense/dense.ply")
fig, ax = plt.subplots(1, 2, figsize=(W, 1.75))
for a, (P, Cc, ttl, s_) in zip(ax, ((sp, sc, f"(a) sparse: {len(sp):,} points", 0.35), (dp, dc, f"(b) dense: {len(dp):,} points", 0.04))):
    Q = P @ M16.T
    m = viz.clean_mask(Q, 2.5)
    idx = np.flatnonzero(m)
    if len(idx) > 220000:
        idx = np.random.default_rng(0).choice(idx, 220000, replace=False)
    a.scatter(Q[idx, 0], Q[idx, 2], c=Cc[idx] / 255.0, s=s_, linewidths=0, rasterized=True)
    a.set_aspect("equal"); a.axis("off"); a.set_title(ttl, fontsize=6.5)
ax[0].set_xlim(ax[1].get_xlim()); ax[0].set_ylim(ax[1].get_ylim())
fig.tight_layout(pad=0.3)
save(fig, "fig7_dense.png")

# ------------------------------------------------------------------------------------------- Fig 8: failures
fol = cv2.imread("results/person_hall/B/matches/matches_IMG_1045_IMG_1048_c_ransac.jpg")[34:, :, ::-1]
fig = plt.figure(figsize=(W, 3.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.35, wspace=0.38)
a0 = fig.add_subplot(gs[0, :]); a0.imshow(fol); a0.axis("off")
a0.set_title("(a) case B (foliage): all 74 surviving matches are correct, but too few", fontsize=6)
try:
    Mf = np.load("results/person_hall/F/inlier_matrix.npy")
except Exception:
    Mf = None
a1 = fig.add_subplot(gs[1, 0])
if Mf is not None:
    a1.imshow(np.log10(Mf + 1), cmap="viridis")
a1.set_title("(b) loop F: verified inliers\n(log10), image pairs", fontsize=6); a1.set_xlabel("image"); a1.set_ylabel("image"); a1.tick_params(length=2)
dg = pd.read_csv("results/gpu/diag_F3_dls_cameras.csv")
a2 = fig.add_subplot(gs[1, 1])
a2.scatter(dg.idx, dg.rot_err_deg, c=np.where(dg.correct, "tab:green", "tab:red"), s=5, linewidths=0)
a2.set_title("(c) learned loop F3: rotation\nerror along the chain", fontsize=6); a2.set_xlabel("camera index (sampling order)"); a2.set_ylabel("rotation error (deg)")
a2.tick_params(length=2)
save(fig, "fig8_failures.png")

# ------------------------------------------------------------------------------------------- Fig 9: bake-off + COLMAP overlay
bs = pd.read_csv("results/gpu/bakeoff_summary.csv")
groups = [("foliage (case B)", "foliage"), ("scale change (case D)", "scale\nchange"), ("sun flare (case C)", "sun\nflare"), ("F3 weak link", "loop weak\nlinks")]
meth = [("sift_bf", "RootSIFT + ratio", "tab:gray"), ("disk_nn", "DISK + ratio", "tab:orange"), ("disk_lg", "DISK + LightGlue", "tab:green")]
fig, ax = plt.subplots(1, 2, figsize=(W, 1.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
a = ax[0]
for k, (m_, lab, c_) in enumerate(meth):
    vals = [float(bs[(bs.group == g) & (bs.method == m_)].median_correct.iloc[0]) for g, _ in groups]
    a.bar(np.arange(len(groups)) + (k - 1) * 0.27, np.array(vals) + 1, 0.27, color=c_, label=lab)
a.set_yscale("log"); a.set_xticks(np.arange(len(groups))); a.set_xticklabels([g for _, g in groups], fontsize=5.3)
a.set_ylabel("median correct inliers per pair"); a.legend(fontsize=4.8, loc="upper left"); a.tick_params(length=2)
a.set_title("(a) matcher bake-off", fontsize=6.5)
cmp_ = compare(rec, pc, include_mask=True)
keepm, sim3 = cmp_["_keep"], cmp_["_sim3"]
s_, R_, t_ = sim3
ours = s_ * pc["xyz"] @ R_.T + t_
colm = south.gt_points[keepm]
mk = viz.clean_mask(ours, 3.0)
c_, r_ = viz.robust_bounds(np.vstack([ours[mk], colm]), lo=1, hi=99, pad=0.05)
b = ax[1]
b.scatter(colm[:, 0], colm[:, 2], s=0.25, c="tab:blue", alpha=0.5, linewidths=0, label=f"COLMAP ({len(colm):,})", rasterized=True)
b.scatter(ours[mk][:, 0], ours[mk][:, 2], s=0.25, c="tab:red", alpha=0.5, linewidths=0, label=f"ours ({int(mk.sum()):,})", rasterized=True)
b.set_xlim(c_[0] - r_, c_[0] + r_); b.set_ylim(c_[2] - r_, c_[2] + r_); b.set_aspect("equal"); b.invert_yaxis()
b.legend(markerscale=6, fontsize=4.8, loc="lower right"); b.tick_params(length=2); b.set_title("(b) vs COLMAP (plan view)", fontsize=6.5)
fig.tight_layout(pad=0.3)
save(fig, "fig9_learned_colmap.png")
print("all figures done")
