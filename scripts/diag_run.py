"""Generic diagnosis of ONE reconstruction against COLMAP ground truth: which cameras and which verified pairs are wrong?

    python scripts/diag_run.py --dataset south-building --start 40 --count 8 --stride 6 --detector disk --matcher lightglue --tag ladder6_dl
Prints / saves results/gpu/diag_<tag>_{cameras,pairs}.csv and a summary json. A camera is 'correct' if its position error
is < 2 % of the trajectory extent AND its rotation error < 5 deg after robust Sim(3) alignment; a pair is 'correct' if its
relative rotation is within 3 deg and translation direction within 20 deg of the ground truth.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, evaluate as Ev, geometry as G
from sfm.reconstruction import Reconstruction, SfMConfig

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="south-building"); ap.add_argument("--start", type=int, default=40)
ap.add_argument("--count", type=int, default=8); ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--detector", default="rootsift"); ap.add_argument("--matcher", default="bf")
ap.add_argument("--window", type=int, default=None); ap.add_argument("--ba_every", type=int, default=1)
ap.add_argument("--min_pair", type=int, default=None, help="sets BOTH min_pair_inliers and min_pair_use")
ap.add_argument("--tag", required=True)
a = ap.parse_args()

ds = D.load_dataset(a.dataset)
names = D.select_images(ds, a.start, a.count, a.stride)
cfg = SfMConfig(detector=a.detector, matcher=a.matcher, window=a.window, ba_every=a.ba_every)
if a.min_pair:
    cfg.min_pair_inliers = cfg.min_pair_use = a.min_pair
rec = Reconstruction(ds, names, cfg, cache_dir="results/cache", verbose=False).run()
pc = rec.point_cloud(with_colors=False)
gt = Ev.evaluate_against_gt(rec, pc)
s, R, t = gt["_sim3"]
C_gt_all = np.array([G.camera_center(*ds.gt_poses[n]) for n in names])
extent = np.linalg.norm(C_gt_all - C_gt_all.mean(0), axis=1).max()

rows = []
for i in sorted(rec.cams):
    Re, te = rec.cams[i]; Rg, tg = ds.gt_poses[names[i]]
    pos = np.linalg.norm(s * G.camera_center(Re, te) @ R.T + t - G.camera_center(Rg, tg)) / extent
    rot = G.rotation_angle_deg(Re @ R.T @ Rg.T)
    rows.append(dict(idx=i, image=names[i], pos_err_rel=pos, rot_err_deg=rot, correct=bool(pos < 0.02 and rot < 5)))
cam = pd.DataFrame(rows)

prow = []
for (i, j), m in sorted(rec.matches.items()):
    if m["n_inl"] < rec.cfg.min_pair_inliers:
        continue
    p1, p2 = rec.kps[i][m["i1"]], rec.kps[j][m["i2"]]
    Rr, tr, _, _ = G.recover_pose_cheirality(m["E"], rec.K, p1, p2)
    (Ri, ti), (Rj, tj) = ds.gt_poses[names[i]], ds.gt_poses[names[j]]
    Rg = Rj @ Ri.T; tg = tj - Rg @ ti
    rot = G.rotation_angle_deg(Rr @ Rg.T)
    tdir = np.degrees(np.arccos(np.clip(tr @ tg / np.linalg.norm(tg), -1, 1)))
    ok = Ev.match_correctness(rec, i, j, m["i1"], m["i2"], thr_px=1.5)
    prow.append(dict(i=i, j=j, gap=j - i, inliers=m["n_inl"], wrong_match_frac=float(1 - ok.mean()), rot_err_deg=float(rot),
                     t_dir_err_deg=float(tdir), pair_correct=bool(rot < 3 and tdir < 20)))
pr = pd.DataFrame(prow)

out = "results/gpu"
cam.to_csv(f"{out}/diag_{a.tag}_cameras.csv", index=False); pr.to_csv(f"{out}/diag_{a.tag}_pairs.csv", index=False)
summ = dict(tag=a.tag, registered=int(len(cam)), images=len(names), cameras_correct=int(cam.correct.sum()),
            cameras_wrong=int((~cam.correct).sum()), verified_pairs=int(len(pr)),
            pairs_correct=int(pr.pair_correct.sum()) if len(pr) else 0,
            pairs_wrong=int((~pr.pair_correct).sum()) if len(pr) else 0,
            median_wrong_match_frac_of_wrong_pairs=float(pr[~pr.pair_correct].wrong_match_frac.median()) if (len(pr) and (~pr.pair_correct).any()) else None,
            median_inliers_wrong_pairs=float(pr[~pr.pair_correct].inliers.median()) if (len(pr) and (~pr.pair_correct).any()) else None,
            median_inliers_correct_pairs=float(pr[pr.pair_correct].inliers.median()) if (len(pr) and pr.pair_correct.any()) else None,
            align_collinear_fix=gt.get("align_collinear_fix"))
json.dump(summ, open(f"{out}/diag_{a.tag}.json", "w"), indent=2, default=float)
print(json.dumps(summ, indent=1, default=float))
print(cam.round(3).to_string(index=False))
