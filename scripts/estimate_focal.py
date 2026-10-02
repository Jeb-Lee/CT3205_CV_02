"""Estimate the focal length of an uncalibrated set from fundamental matrices (no reconstruction needed).

For the true calibration K, E = K^T F K is an essential matrix: its two non-zero singular values are EQUAL. For a grid of
focal lengths f (square pixels, principal point at the image centre) we measure c(f) = (s1 - s2) / (s1 + s2) of K(f)^T F K(f)
over many image pairs and take the f that minimises the median c. Robust (RANSAC F, many pairs), independent of the 3D model.

    python scripts/estimate_focal.py --set south-building --n_pairs 40        # validation: COLMAP f is known
    python scripts/estimate_focal.py --set yellow_sign --n_pairs 60
Writes results/own/focal_<set>.json and .png
"""
import argparse, json, os, re, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D


def focal_cost_curve(F, cx, cy, f_grid):
    out = np.empty(len(f_grid))
    for k, f in enumerate(f_grid):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        s = np.linalg.svd(K.T @ F @ K, compute_uv=False)
        out[k] = (s[0] - s[1]) / (s[0] + s[1] + 1e-12)
    return out


def estimate(ds, n_pairs=40, gap=2, max_dim=1600, nfeatures=6000, span=0.35, seed=0, verbose=True):
    names = ds.names
    n = len(names)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(max(n - gap, 1), min(n_pairs, max(n - gap, 1)), replace=False))
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    cache = {}

    def feats(i):
        if i not in cache:
            img, K = D.load_image(ds, names[i], max_dim)
            k, d = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            d = d / (np.abs(d).sum(1, keepdims=True) + 1e-7)
            cache[i] = (np.array([p.pt for p in k], np.float32), np.sqrt(d).astype(np.float32), K)
        return cache[i]

    _, _, K0 = feats(int(idx[0]))
    f0, cx, cy = float(K0[0, 0]), float(K0[0, 2]), float(K0[1, 2])
    f_grid = f0 * np.linspace(1 - span, 1 + span, 141)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    curves, used = [], []
    for i in idx:
        j = int(i) + gap
        (p1, d1, _), (p2, d2, _) = feats(int(i)), feats(j)
        m = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
        if len(good) < 80:
            continue
        a1 = p1[[g.queryIdx for g in good]]; a2 = p2[[g.trainIdx for g in good]]
        F, mask = cv2.findFundamentalMat(a1, a2, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
        if F is None or mask is None or mask.sum() < 60 or F.shape != (3, 3):
            continue
        curves.append(focal_cost_curve(F, cx, cy, f_grid)); used.append(int(mask.sum()))
    C = np.array(curves)
    med = np.median(C, axis=0)
    best = int(np.argmin(med))
    per_pair = f_grid[np.argmin(C, axis=1)]
    res = dict(n_pairs_used=len(C), f0_exif=f0, f_est=float(f_grid[best]), factor=float(f_grid[best] / f0),
               per_pair_median=float(np.median(per_pair)), per_pair_q25=float(np.percentile(per_pair, 25)), per_pair_q75=float(np.percentile(per_pair, 75)),
               cx=cx, cy=cy, median_inliers=int(np.median(used)))
    if verbose:
        print(f"{ds.name}: {len(C)} pairs; f_exif={f0:.1f}  f_est={res['f_est']:.1f}  (x{res['factor']:.3f}); per-pair median {res['per_pair_median']:.1f} "
              f"[IQR {res['per_pair_q25']:.1f}-{res['per_pair_q75']:.1f}]", flush=True)
    return res, f_grid, med, C


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--n_pairs", type=int, default=40); ap.add_argument("--gap", type=int, default=2)
    a = ap.parse_args()
    ds = D.load_dataset(a.set)
    res, fg, med, C = estimate(ds, a.n_pairs, a.gap)
    tag = re.sub(r"[^A-Za-z0-9_]", "_", a.set)
    os.makedirs("results/own", exist_ok=True)
    json.dump(res, open(f"results/own/focal_{tag}.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for c in C[:20]:
        ax.plot(fg, c, color="0.8", lw=0.6)
    ax.plot(fg, med, "b-", lw=1.5, label="median over pairs"); ax.axvline(res["f_est"], color="r", ls=":", label=f"estimate {res['f_est']:.0f}")
    ax.axvline(res["f0_exif"], color="g", ls="--", label=f"EXIF {res['f0_exif']:.0f}")
    ax.set_ylim(0, min(0.5, float(np.percentile(C, 90)))); ax.set_xlabel("focal length (px, working res.)"); ax.set_ylabel("(s1-s2)/(s1+s2)")
    ax.legend(fontsize=6); ax.set_title(f"{a.set}: focal from F matrices", fontsize=8); fig.tight_layout(); fig.savefig(f"results/own/focal_{tag}.png", dpi=130)
