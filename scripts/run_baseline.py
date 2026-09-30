"""Baseline multi-view reconstruction + all report artefacts.

    python scripts/run_baseline.py --dataset south-building --start 0 --count 12 --stride 2 --out results/baseline
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, viz, evaluate as Ev, geometry as G
from sfm.reconstruction import Reconstruction, SfMConfig


def two_view_demo(rec, out):
    """Report material for 1.3: the four candidate poses of E and the cheirality selection, vs OpenCV."""
    i, j = rec.init_pair
    m = rec.matches[(i, j)]
    p1, p2 = rec.kps[i][m["i1"]], rec.kps[j][m["i2"]]
    E = m["E"]
    s = np.linalg.svd(E, compute_uv=False)
    rows = []
    xn1, xn2 = G.to_normalized(p1, rec.K), G.to_normalized(p2, rec.K)
    for k, (R, t) in enumerate(G.decompose_essential(E)):
        X = G.triangulate_dlt(np.eye(3), np.zeros(3), R, t, xn1, xn2)
        front = int(np.sum((G.depth(np.eye(3), np.zeros(3), X) > 0) & (G.depth(R, t, X) > 0)))
        rows.append(dict(candidate=k + 1, rot_angle_deg=float(G.rotation_angle_deg(R)), t=t.tolist(),
                         points_in_front_of_both=front, n=len(p1)))
    R, t, _, X = G.recover_pose_cheirality(E, rec.K, p1, p2)
    _, Rcv, tcv, _ = cv2.recoverPose(E, p1, p2, rec.K)
    # DLT vs OpenCV triangulation
    P1 = rec.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = rec.K @ np.hstack([R, t.reshape(3, 1)])
    Xcv = cv2.triangulatePoints(P1, P2, p1.T.astype(float), p2.T.astype(float))
    Xcv = (Xcv[:3] / Xcv[3]).T
    info = dict(pair=[rec.names[i], rec.names[j]], singular_values_of_E=s.tolist(),
                candidates=rows, chosen_rot_deg=float(G.rotation_angle_deg(R)), chosen_t=t.tolist(),
                agrees_with_opencv_recoverPose=bool(np.allclose(R, Rcv, atol=1e-3)),
                cv_rot_err_deg=float(G.rotation_angle_deg(R @ Rcv.T)),
                cv_t_cos=float(abs(t @ tcv.ravel())),
                dlt_vs_opencv_max_abs_diff=float(np.abs(X - Xcv).max()),
                epipolar_check_max_xt_E_x=float(np.abs(np.sum(np.hstack([xn2, np.ones((len(xn2), 1))]) *
                                                              (np.hstack([xn1, np.ones((len(xn1), 1))]) @ E.T), 1)).max()))
    json.dump(info, open(os.path.join(out, "two_view_demo.json"), "w"), indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="south-building")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--detector", default="rootsift")
    ap.add_argument("--max_dim", type=int, default=1600)
    ap.add_argument("--nfeatures", type=int, default=8000)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--no_ba", action="store_true")
    ap.add_argument("--video", action="store_true", help="also render turntable mp4")
    ap.add_argument("--out", default="results/baseline")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    ds = D.load_dataset(a.dataset)
    names = D.select_images(ds, a.start, a.count, a.stride)
    print("images:", names)
    cfg = SfMConfig(detector=a.detector, max_dim=a.max_dim, nfeatures=a.nfeatures, window=a.window, use_ba=not a.no_ba)
    rec = Reconstruction(ds, names, cfg, cache_dir=os.path.join("results", "cache")).run()
    pc = rec.point_cloud()

    st = rec.stats()
    gt = Ev.evaluate_against_gt(rec, pc)
    sim = gt.pop("_sim3", None)
    st.update(gt)
    if sim is not None:   # Umeyama alignment ours -> COLMAP frame (used by view_open3d.py --gt)
        st["sim3"] = dict(s=float(sim[0]), R=sim[1].tolist(), t=sim[2].tolist())
    st["config"] = cfg.__dict__
    st["images"] = names
    st["log_events"] = rec.log["events"]
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in st.items() if k not in ("log_events", "images", "config")}, indent=1, default=float))

    # ---- point cloud + cameras
    viz.write_ply(os.path.join(a.out, "cloud.ply"), pc["xyz"], pc["rgb"])
    viz.write_cameras_ply(os.path.join(a.out, "cameras.ply"), rec, scale=0.15 * viz.robust_bounds(pc["xyz"])[1])
    viz.plot_scene(rec, pc, os.path.join(a.out, "scene_views.png"))
    viz.error_histogram(rec, os.path.join(a.out, "reproj_hist.png"))

    # ---- match stages for the strongest pair + init pair
    i, j = rec.init_pair
    print("match stages", viz.save_match_stages(rec, i, j, os.path.join(a.out, "matches")))
    # neighbouring pair (i, i+1) as a second example
    print("two-view demo", json.dumps(two_view_demo(rec, a.out), indent=1))

    # ---- reprojection examples for 3 cameras
    regs = sorted(rec.cams)
    for im in [regs[0], regs[len(regs) // 2], regs[-1]]:
        print("reproj", rec.names[im], viz.reprojection_figure(rec, im, os.path.join(a.out, f"reproj_{rec.names[im][:-4]}.jpg")))

    if a.video:
        viz.turntable_video(rec, pc, os.path.join(a.out, "turntable.mp4"))
    print("saved to", a.out)


if __name__ == "__main__":
    main()
