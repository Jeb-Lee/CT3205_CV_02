"""Diagnostic: print the verified-inlier matrix for a list of images (which pairs actually link up?)."""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D
from sfm.reconstruction import Reconstruction, SfMConfig

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="south-building"); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=16); ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--ratio", type=float, default=0.8); ap.add_argument("--nfeatures", type=int, default=8000)
ap.add_argument("--max_dim", type=int, default=1600); ap.add_argument("--thr", type=float, default=1.0)
ap.add_argument("--detector", default="rootsift")
a = ap.parse_args()
ds = D.load_dataset(a.dataset)
names = D.select_images(ds, a.start, a.count, a.stride)
cfg = SfMConfig(detector=a.detector, ratio=a.ratio, nfeatures=a.nfeatures, max_dim=a.max_dim, ransac_thr=a.thr)
r = Reconstruction(ds, names, cfg, cache_dir="results/cache", verbose=False)
r.compute_features(); r.match_all()
n = len(names)
M = np.zeros((n, n), int)
for (i, j), m in r.matches.items():
    M[i, j] = M[j, i] = m["n_inl"]
print("images:", [x[-8:-4] for x in names])
print(M)
