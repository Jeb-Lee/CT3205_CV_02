"""Matcher bake-off on hard image pairs, scored against COLMAP ground-truth geometry.

Methods
  sift_bf   : RootSIFT 8000 + ratio 0.8 + mutual check (CPU brute force)         - our baseline
  disk_nn   : DISK 8000 features + the SAME ratio/mutual matcher on the GPU       - isolates the benefit of learned FEATURES
  disk_lg   : DISK 8000 features + LightGlue                                      - adds the learned MATCHER

Per pair/method: raw matches, matches surviving RANSAC-E (1 px), how many of those are geometrically correct under the
ground-truth poses (Sampson < 1.5 px) and matching time. Pair sets: easy / viewpoint ladder (south-building), foliage,
scale change, sun flare, repetitive arches (Person Hall) and the WEAK LINKS of the whole-loop run F3 (adjacent samples that
had < 25 verified SIFT inliers and therefore cut the loop).

    python scripts/matcher_bakeoff.py
"""
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, features as F, matching as M, geometry as G, gpu_matching as GM, viz

OUT = os.path.join("results", "gpu")
os.makedirs(OUT, exist_ok=True)
MAXDIM = 1600
_cache = {}


def get(ds, name, dl):
    key = (ds.name, name)
    if key not in _cache:
        img, K = D.load_image(ds, name, MAXDIM)
        kp_s, d_s = F.extract(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "rootsift", 8000)
        kp_d, d_d = dl.extract(img)
        _cache[key] = dict(img=img, K=K, sift=(kp_s, d_s), disk=(kp_d, d_d))
    return _cache[key]


def gt_correct(ds, na, nb, K, p1, p2, thr=1.5):
    (Ri, ti), (Rj, tj) = ds.gt_poses[na], ds.gt_poses[nb]
    Rij = Rj @ Ri.T
    tij = tj - Rij @ ti
    Kinv = np.linalg.inv(K)
    Fm = Kinv.T @ (G.skew(tij) @ Rij) @ Kinv
    return np.sqrt(G.sampson_error(Fm, p1, p2)) < thr


def run_methods(ds, na, nb, dl):
    A, B = get(ds, na, dl), get(ds, nb, dl)
    K = A["K"]
    size = (2 * K[0, 2], 2 * K[1, 2])
    rows, vis = [], {}
    for method in ("sift_bf", "disk_nn", "disk_lg"):
        if method == "sift_bf":
            (k1, d1), (k2, d2) = A["sift"], B["sift"]
            t0 = time.time(); i1, i2, _ = M.match_pair(d1, d2, cv2.NORM_L2, 0.8, True); dt = time.time() - t0
        elif method == "disk_nn":
            (k1, d1), (k2, d2) = A["disk"], B["disk"]
            GM.sync(); t0 = time.time(); i1, i2, _ = GM.match_pair_gpu(d1, d2, 0.8, True); GM.sync(); dt = time.time() - t0
        else:
            (k1, d1), (k2, d2) = A["disk"], B["disk"]
            GM.sync(); t0 = time.time(); i1, i2, _ = dl.match(k1, d1, k2, d2, size, size); GM.sync(); dt = time.time() - t0
        n_raw = len(i1)
        ok_all = gt_correct(ds, na, nb, K, k1[i1], k2[i2]) if n_raw else np.zeros(0, bool)
        E, mask = M.verify_essential(k1[i1], k2[i2], K, 1.0) if n_raw >= 8 else (None, np.zeros(n_raw, bool))
        n_inl = int(mask.sum())
        n_inl_correct = int((ok_all & mask).sum()) if n_raw else 0
        rows.append(dict(method=method, matches=int(n_raw), precision_raw=float(ok_all.mean()) if n_raw else np.nan,
                         ransac_inliers=n_inl, inliers_correct=n_inl_correct,
                         seconds=dt))
        vis[method] = (k1[i1[mask]], k2[i2[mask]], ok_all[mask] if n_raw else np.zeros(0, bool))
    return rows, vis, (A["img"], B["img"])


def main():
    dl = GM.DiskLightGlue(8000)
    south, hall = D.load_dataset("south-building"), D.load_dataset("person-hall")
    pairs = []
    sn = south.names
    for k in (2, 4, 6, 8, 10, 14):                      # viewpoint ladder from the same start image
        pairs.append(("south ladder", south, sn[40], sn[40 + k], f"gap {k}"))
    nm = lambda n: f"IMG_{n}.JPG"
    pairs += [("foliage (case B)", hall, nm(1045), nm(1048), "B 1045-1048"),
              ("foliage (case B)", hall, nm(1051), nm(1057), "B 1051-1057"),
              ("foliage (case B)", hall, nm(1060), nm(1066), "B 1060-1066"),
              ("scale change (case D)", hall, nm(1309), nm(1313), "D 1309-1313"),
              ("scale change (case D)", hall, nm(1313), nm(1317), "D 1313-1317"),
              ("sun flare (case C)", hall, nm(1222), nm(1226), "C 1222-1226")]
    # weak links of the whole-loop run F3: adjacent samples (stride 3) that had < 25 verified SIFT inliers
    f3 = np.load("results/person_hall/F3/inlier_matrix.npy")
    f3_names = D.select_images(hall, 0, 110, 3)
    weak = [(i, i + 1, int(f3[i, i + 1])) for i in range(109) if f3[i, i + 1] < 25]
    print(f"F3 weak adjacent links (<25 SIFT inliers): {len(weak)} of 109")
    rng = np.random.default_rng(0)
    for i, j, n0 in [weak[k] for k in rng.choice(len(weak), min(len(weak), 14), replace=False)]:
        pairs.append(("F3 weak link", hall, f3_names[i], f3_names[j], f"F3 {i}-{j} (SIFT {n0})"))

    out, samples = [], {}
    for group, ds, na, nb, label in pairs:
        try:
            rows, vis, imgs = run_methods(ds, na, nb, dl)
        except Exception as e:
            print("  failed", label, e); continue
        for r in rows:
            r.update(group=group, pair=label, a=na, b=nb)
            out.append(r)
        samples[label] = (vis, imgs)
        s = {r["method"]: r["inliers_correct"] for r in rows}
        print(f"{group:24s} {label:22s} correct inliers: sift {s['sift_bf']:5d} | disk+nn {s['disk_nn']:5d} | disk+LG {s['disk_lg']:5d}", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "bakeoff.csv"), index=False)

    # ------------------------------------------------ summary per group
    g = df.groupby(["group", "method"]).agg(pairs=("pair", "nunique"), median_inliers=("ransac_inliers", "median"),
                                            median_correct=("inliers_correct", "median"),
                                            median_precision_raw=("precision_raw", "median"),
                                            median_seconds=("seconds", "median")).round(3)
    g.to_csv(os.path.join(OUT, "bakeoff_summary.csv"))
    print("\n", g.to_string())
    w = df[df.group == "F3 weak link"].pivot(index="pair", columns="method", values="inliers_correct")
    thr = 25
    print(f"\nF3 weak links bridged (>= {thr} correct verified matches): " +
          ", ".join(f"{m}: {(w[m] >= thr).sum()}/{len(w)}" for m in ("sift_bf", "disk_nn", "disk_lg")))

    # ------------------------------------------------ figure: correct inliers per pair, 3 methods
    piv = df.pivot_table(index=["group", "pair"], columns="method", values="inliers_correct", sort=False)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(piv)); wd = 0.27
    for k, (m, c, lab) in enumerate((("sift_bf", "tab:gray", "RootSIFT + ratio (CPU)"), ("disk_nn", "tab:orange", "DISK + ratio (GPU)"),
                                     ("disk_lg", "tab:green", "DISK + LightGlue (GPU)"))):
        ax.bar(x + (k - 1) * wd, piv[m].values + 1, wd, color=c, label=lab)
    ax.set_yscale("log"); ax.axhline(25, color="red", ls="--", lw=0.8); ax.text(len(piv) - 0.5, 28, "25 = min. usable", color="red", ha="right", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([p[1] for p in piv.index], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("geometrically correct RANSAC inliers (+1, log)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title("Matcher bake-off against COLMAP ground-truth geometry (higher is better)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bakeoff_inliers.png"), dpi=130); plt.close(fig)

    # ------------------------------------------------ example match figures (SIFT vs LightGlue), colour = GT correct
    for label in ("B 1045-1048", "D 1309-1313"):
        if label not in samples:
            continue
        vis, (ia, ib) = samples[label]
        tiles = []
        for m, ttl in (("sift_bf", "RootSIFT + ratio"), ("disk_lg", "DISK + LightGlue")):
            p1, p2, ok = vis[m]
            col = [(0, 220, 0) if o else (0, 0, 255) for o in ok]
            tiles.append(viz.draw_matches(ia, ib, p1, p2, col, 200, title=f"{label}: {ttl}: {len(p1)} verified matches"))
        cv2.imwrite(os.path.join(OUT, f"bakeoff_example_{label.split()[0]}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("saved to", OUT)


if __name__ == "__main__":
    main()
