"""Per-camera and per-pair failure diagnosis for a Person Hall case (needs COLMAP ground truth).

    python scripts/diag_case.py C

Writes results/person_hall/<case>/diag_*.csv|png|jpg:
  * per-camera position / rotation error after Sim(3) alignment  -> which cameras are wrong
  * per-pair relative-pose error vs GT for every verified pair, and the share of verified matches that violate the
    GT epipolar geometry -> which image pairs carry false (e.g. repetitive-structure) matches
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
sys.path.insert(0, os.path.dirname(__file__))
from sfm import datasets as D, viz, evaluate as Ev, geometry as G
from sfm.reconstruction import Reconstruction, SfMConfig
from run_person_hall import CASES, CACHE, OUT


def main(key):
    title, why, start, count, stride, over = CASES[key]
    ds = D.load_dataset("person-hall")
    names = D.select_images(ds, start, count, stride)
    rec = Reconstruction(ds, names, SfMConfig(**over), cache_dir=CACHE, verbose=False).run()
    out = os.path.join(OUT, key)
    pc = rec.point_cloud(with_colors=False)
    gt = Ev.evaluate_against_gt(rec, pc)
    s, R, t = gt["_sim3"]

    # ---- per camera
    rows = []
    for i in sorted(rec.cams):
        Re, te = rec.cams[i]
        Rg, tg = ds.gt_poses[names[i]]
        Ce = s * G.camera_center(Re, te) @ R.T + t
        Cg = G.camera_center(Rg, tg)
        rows.append(dict(idx=i, image=names[i], registered_order=list(rec.cams).index(i),
                         pos_err=float(np.linalg.norm(Ce - Cg)), rot_err_deg=float(G.rotation_angle_deg(Re @ R.T @ Rg.T))))
    cam = pd.DataFrame(rows)
    cam["pos_err_rel"] = cam.pos_err / (np.linalg.norm(np.array([G.camera_center(*ds.gt_poses[n]) for n in names]) -
                                       np.mean([G.camera_center(*ds.gt_poses[n]) for n in names], 0), axis=1).max())
    cam.to_csv(os.path.join(out, "diag_cameras.csv"), index=False)
    print("\nper-camera error vs GT:\n", cam.round(3).to_string(index=False))

    # ---- per pair
    prow = []
    for (i, j), m in sorted(rec.matches.items()):
        if m["n_inl"] < rec.cfg.min_pair_inliers:
            continue
        p1, p2 = rec.kps[i][m["i1"]], rec.kps[j][m["i2"]]
        Rr, tr, _, _ = G.recover_pose_cheirality(m["E"], rec.K, p1, p2)
        (Ri, ti), (Rj, tj) = ds.gt_poses[names[i]], ds.gt_poses[names[j]]
        Rg = Rj @ Ri.T
        tg = tj - Rg @ ti
        rot = G.rotation_angle_deg(Rr @ Rg.T)
        tdir = np.degrees(np.arccos(np.clip(tr @ tg / np.linalg.norm(tg), -1, 1)))
        ok = Ev.match_correctness(rec, i, j, m["i1"], m["i2"], thr_px=1.5)
        prow.append(dict(i=i, j=j, gap=j - i, inliers=m["n_inl"], wrong_frac=float(1 - ok.mean()),
                         rot_err_deg=float(rot), t_dir_err_deg=float(tdir)))
    pr = pd.DataFrame(prow)
    pr.to_csv(os.path.join(out, "diag_pairs.csv"), index=False)
    bad = pr[(pr.rot_err_deg > 3) | (pr.t_dir_err_deg > 20)]
    print(f"\nverified pairs: {len(pr)}; with wrong relative pose (rot>3deg or t-dir>20deg): {len(bad)}")
    print(pr.sort_values("rot_err_deg", ascending=False).head(10).round(3).to_string(index=False))
    print("\nmedian wrong-match fraction: all pairs %.3f | bad pairs %.3f" % (pr.wrong_frac.median(),
          bad.wrong_frac.median() if len(bad) else float("nan")))

    # ---- figure: pair-error matrices
    n = rec.n
    Mr = np.full((n, n), np.nan); Mw = np.full((n, n), np.nan)
    for _, r in pr.iterrows():
        Mr[int(r.i), int(r.j)] = Mr[int(r.j), int(r.i)] = r.rot_err_deg
        Mw[int(r.i), int(r.j)] = Mw[int(r.j), int(r.i)] = r.wrong_frac
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a_, M, ttl, cm, vmax in ((ax[0], Mr, "relative-rotation error of pair vs GT (deg)", "magma", 20),
                                 (ax[1], Mw, "share of verified matches violating GT epipolar geometry", "magma", 1)):
        im = a_.imshow(M, cmap=cm, vmin=0, vmax=vmax); a_.set_title(ttl, fontsize=10); a_.set_xlabel("image"); a_.set_ylabel("image")
        fig.colorbar(im, ax=a_)
    fig.suptitle(f"case {key}: {title}  (white = pair not verified)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "diag_pair_errors.png"), dpi=120); plt.close(fig)

    # ---- worst pair, matches coloured by GT correctness
    if len(pr):
        w = pr.sort_values(["wrong_frac", "inliers"], ascending=False).iloc[0]
        i, j = int(w.i), int(w.j)
        img1, _ = D.load_image(ds, names[i], rec.cfg.max_dim)
        img2, _ = D.load_image(ds, names[j], rec.cfg.max_dim)
        v1, v2 = rec.pair_matches(i, j)
        ok = Ev.match_correctness(rec, i, j, v1, v2, thr_px=1.5)
        col = [(0, 220, 0) if o else (0, 0, 255) for o in ok]
        canvas = viz.draw_matches(img1, img2, rec.kps[i][v1], rec.kps[j][v2], col, 250,
                                  title=f"case {key}: pair ({names[i]}, {names[j]}) verified matches, {100*w.wrong_frac:.0f}% violate GT geometry (red)")
        cv2.imwrite(os.path.join(out, "diag_worst_pair.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("worst pair figure:", names[i], names[j], f"{100*w.wrong_frac:.0f}% wrong")


if __name__ == "__main__":
    main(sys.argv[1])
