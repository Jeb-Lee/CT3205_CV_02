"""GPU sanity + speed check: (1) GPU brute-force matcher == CPU matcher? how much faster? (2) DISK + LightGlue run + speed.

    python scripts/gpu_bench.py
Writes results/gpu/gpu_bench.json
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, features as F, matching as M, gpu_matching as GM

os.makedirs("results/gpu", exist_ok=True)
res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)

ds = D.load_dataset("south-building")
names = D.select_images(ds, 40, 3, 2)
imgs = [D.load_image(ds, n, 1600) for n in names]
K = imgs[0][1]

# ---------------------------------------------------------------- (1) classical matcher: CPU vs GPU
feats = [F.extract(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "rootsift", 8000) for im, _ in imgs]
(k1, d1), (k2, d2) = feats[0], feats[1]
a1, a2, sa = M.match_pair(d1, d2, cv2.NORM_L2, 0.8, True)
b1, b2, sb = GM.match_pair_gpu(d1, d2, 0.8, True)
set_a, set_b = set(zip(a1.tolist(), a2.tolist())), set(zip(b1.tolist(), b2.tolist()))
res["bf_cpu_matches"], res["bf_gpu_matches"] = len(set_a), len(set_b)
res["identical_matches"] = len(set_a & set_b)
res["only_cpu"], res["only_gpu"] = len(set_a - set_b), len(set_b - set_a)
cpu_s, gpu_s = GM.bench_matching(d1, d2)
res["match_seconds_cpu"], res["match_seconds_gpu"], res["speedup"] = cpu_s, gpu_s, cpu_s / gpu_s
print(f"[classical] CPU {len(set_a)} matches in {cpu_s:.3f}s | GPU {len(set_b)} matches in {gpu_s:.4f}s | "
      f"identical {res['identical_matches']} | speedup x{cpu_s / gpu_s:.1f}")

# ---------------------------------------------------------------- (2) learned: DISK + LightGlue
t0 = time.time()
dl = GM.DiskLightGlue(8000)
res["model_load_seconds"] = time.time() - t0
print(f"[learned] models loaded in {res['model_load_seconds']:.1f}s (weights downloaded on first run)")
tk = []
out = []
for im, _ in imgs[:2]:
    GM.sync(); t0 = time.time()
    kp, de = dl.extract(im)
    GM.sync(); tk.append(time.time() - t0); out.append((kp, de))
(kk1, dd1), (kk2, dd2) = out
size = (2 * K[0, 2], 2 * K[1, 2])
dl.match(kk1, dd1, kk2, dd2, size, size); GM.sync()      # warm-up
GM.sync(); t0 = time.time()
m1, m2, st = dl.match(kk1, dd1, kk2, dd2, size, size)
GM.sync(); t_match = time.time() - t0
E, mask = M.verify_essential(kk1[m1], kk2[m2], K, 1.0)
E2, mask2 = M.verify_essential(k1[a1], k2[a2], K, 1.0)
res.update(disk_keypoints=[len(kk1), len(kk2)], disk_extract_seconds=tk, lightglue_seconds=t_match,
           lightglue_matches=int(len(m1)), lightglue_ransac_inliers=int(mask.sum()),
           rootsift_matches=int(len(a1)), rootsift_ransac_inliers=int(mask2.sum()))
print(f"[learned] DISK {len(kk1)} kp, extract {tk[1]:.3f}s | LightGlue {len(m1)} matches in {t_match:.3f}s | "
      f"RANSAC inliers {mask.sum()}  (RootSIFT+ratio: {len(a1)} matches, {mask2.sum()} inliers)")
json.dump(res, open("results/gpu/gpu_bench.json", "w"), indent=2, default=float)
