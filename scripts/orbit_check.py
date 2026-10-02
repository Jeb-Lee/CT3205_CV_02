"""GT-free geometry checks of an orbit reconstruction: does the camera path close, and is it a consistent circle?

    python scripts/orbit_check.py                 # all sets in results/own
For each reconstruction (rec.pkl): fit a plane to the camera centres, fit a circle in that plane and report
  * radius and its relative spread (a true orbit at constant distance has a small spread)
  * closure: distance between the first and last camera divided by the median neighbour spacing (a closed 360 deg
    orbit gives ~1; drift or an open arc gives >> 1)
  * angular coverage of the orbit and the median step between neighbouring cameras
Writes results/own/orbit_check.json
"""
import json, os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import geometry as G


def circle_fit(P2):
    A = np.c_[2 * P2, np.ones(len(P2))]
    b = (P2 ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:2]; r = np.sqrt(sol[2] + c @ c)
    return c, r


def analyse(centers, label):
    C = np.asarray(centers)
    mu = C.mean(0)
    U, S, Vt = np.linalg.svd(C - mu)
    P = (C - mu) @ Vt[:2].T                       # coordinates in the best-fit plane
    off = (C - mu) @ Vt[2]                         # distance from the plane
    c, r = circle_fit(P)
    rad = np.linalg.norm(P - c, axis=1)
    ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    out = dict(label=label, n=len(C), radius=float(r), radius_rel_std=float(rad.std() / rad.mean()), plane_offset_rel_std=float(off.std() / rad.mean()),
               angular_coverage_deg=float(abs(ang[-1] - ang[0]) * 180 / np.pi), median_step=float(np.median(steps)),
               closure_ratio=float(np.linalg.norm(C[-1] - C[0]) / np.median(steps)), max_step_over_median=float(steps.max() / np.median(steps)))
    return out


res = []
for tag in ["yellow_all", "fire", "snacks", "bottle_classic", "bottle_learned"]:
    pk = f"results/own/{tag}/rec.pkl"
    if not os.path.exists(pk):
        continue
    rec = pickle.load(open(pk, "rb"))
    order = sorted(rec.cams)
    C = np.array([G.camera_center(*rec.cams[i]) for i in order])
    if tag == "yellow_all":                         # two rings: level (first 82 frames) and high (last 37) analysed separately
        n_lvl = sum(1 for i in order if "Hori" in rec.names[i])
        res.append(analyse(C[:n_lvl], "yellow level ring")); res.append(analyse(C[n_lvl:], "yellow high ring"))
        res[-2]["radius_ratio_high_over_level"] = res[-1]["radius"] / res[-2]["radius"]
    else:
        res.append(analyse(C, tag))
for r in res:
    print(f"{r['label']:20s} n={r['n']:3d}  radius {r['radius']:.2f}  radius spread {100*r['radius_rel_std']:.1f}%  height spread {100*r['plane_offset_rel_std']:.1f}%  "
          f"coverage {r['angular_coverage_deg']:.0f} deg  step {r['median_step']:.2f} (max x{r['max_step_over_median']:.1f})  closure first-last = {r['closure_ratio']:.1f} steps")
json.dump(res, open("results/own/orbit_check.json", "w"), indent=1)
