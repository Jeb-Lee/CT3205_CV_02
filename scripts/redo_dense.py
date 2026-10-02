"""Object-focused dense reconstruction of an orbit capture (re-uses the saved sparse model rec.pkl).

    python scripts/redo_dense.py --tag yellow_all --gap 3 --crop 0.35 --axial 0.6
Differences to the default dense stage: wider stereo baseline (pair_gap, because orbit frames are only ~5 deg apart), and the
fusion is restricted to a cylinder around the orbit axis (the object) with a voxel sized from that region.
"""
import argparse, os, pickle, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import geometry as G, viz
from sfm.dense import DenseConfig, dense_reconstruct

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--gap", type=int, default=3)
ap.add_argument("--crop", type=float, default=0.35); ap.add_argument("--axial", type=float, default=0.6)
ap.add_argument("--support", type=int, default=2); ap.add_argument("--voxel_frac", type=float, default=1 / 300)
ap.add_argument("--uniqueness", type=int, default=10); ap.add_argument("--block", type=int, default=5)
ap.add_argument("--out_name", default="dense_object.ply")
a = ap.parse_args()
d = os.path.join("results", "own", a.tag)
rec = pickle.load(open(os.path.join(d, "rec.pkl"), "rb"))
Cc = np.array([G.camera_center(*rec.cams[i]) for i in sorted(rec.cams)])
mu = Cc.mean(0); _, _, Vt = np.linalg.svd(Cc - mu)
P2 = (Cc - mu) @ Vt[:2].T
sol = np.linalg.lstsq(np.c_[2 * P2, np.ones(len(P2))], (P2 ** 2).sum(1), rcond=None)[0]
c2, R = sol[:2], float(np.sqrt(sol[2] + sol[:2] @ sol[:2]))
roi = lambda P: (np.linalg.norm((P - mu) @ Vt[:2].T - c2, axis=1) < a.crop * R) & (np.abs((P - mu) @ Vt[2]) < a.axial * R)
cfg = DenseConfig(pair_gap=a.gap, min_support=a.support, voxel_frac=a.voxel_frac, uniqueness=a.uniqueness, block=a.block)
dense = dense_reconstruct(rec, cfg, verbose=False, roi=roi)
viz.write_ply(os.path.join(d, a.out_name), dense["xyz"], dense["rgb"])
info = dict(tag=a.tag, gap=a.gap, crop=a.crop, axial=a.axial, support=a.support, voxel_frac=a.voxel_frac, orbit_radius=R,
            points=int(len(dense["xyz"])), raw_points=dense["info"]["raw_points"], pairs=dense["info"]["n_pairs"], seconds=dense["info"]["seconds"])
json.dump(info, open(os.path.join(d, a.out_name.replace(".ply", ".json")), "w"), indent=1)
print(info)
