"""GPU feature matching: (1) exact GPU port of ratio+mutual brute-force matching, (2) learned DISK features + LightGlue.

(1) `match_pair_gpu` gives the same result as `matching.match_pair` (L2 descriptors) but computes the distance matrix with
    torch.cdist on the GPU - used to measure how much of the runtime is matching.
(2) DISK (Tyszkiewicz et al., NeurIPS 2020; local features learned with policy-gradient on SfM-style supervision) detects
    keypoints and 128-d descriptors; LightGlue (Lindenberger et al., ICCV 2023) is a transformer matcher that attends
    over both images' keypoints jointly and prunes unmatchable points. Pretrained weights are downloaded by kornia on
    first use (DISK 'depth' checkpoint, LightGlue 'disk' checkpoint). No training is done in this project.
"""
from __future__ import annotations

import time

import numpy as np
import torch

_DEV = None


def device():
    global _DEV
    if _DEV is None:
        _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEV


# ----------------------------------------------------------------------------------------------- classical on GPU
@torch.no_grad()
def match_pair_gpu(d1, d2, ratio=0.8, cross_check=True):
    """Ratio test + mutual check on the GPU. d1, d2: float32 (N,D),(M,D). Returns idx1, idx2, stats like match_pair."""
    dev = device()
    a = torch.from_numpy(np.ascontiguousarray(d1)).to(dev)
    b = torch.from_numpy(np.ascontiguousarray(d2)).to(dev)
    dist = torch.cdist(a, b)                                  # (N, M) euclidean
    top2 = dist.topk(2, dim=1, largest=False)
    best, idx = top2.values[:, 0], top2.indices[:, 0]
    ok = best < ratio * top2.values[:, 1]
    i1 = torch.nonzero(ok).squeeze(1)
    i2 = idx[i1]
    stats = dict(n_raw=len(d1), n_ratio=int(len(i1)))
    if cross_check and len(i1):
        back = dist.argmin(dim=0)                             # best partner in image 1 for every keypoint of image 2
        keep = back[i2] == i1
        i1, i2 = i1[keep], i2[keep]
    stats["n_mutual"] = int(len(i1))
    return i1.cpu().numpy(), i2.cpu().numpy(), stats


# ----------------------------------------------------------------------------------------------- learned features
class DiskLightGlue:
    """DISK keypoints/descriptors + LightGlue matching, both on the GPU."""

    def __init__(self, max_keypoints=8000, filter_threshold=0.1):
        from kornia.feature import DISK, LightGlue
        dev = device()
        self.max_kp = max_keypoints
        self.disk = DISK.from_pretrained("depth").to(dev).eval()
        self.lg = LightGlue("disk").to(dev).eval()
        self.lg.conf.filter_threshold = filter_threshold if hasattr(self.lg, "conf") else None

    @torch.no_grad()
    def extract(self, bgr):
        """bgr uint8 image -> keypoints (N,2) xy float32, descriptors (N,128) float32."""
        dev = device()
        x = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255.0)[None].to(dev)
        f = self.disk(x, n=self.max_kp, pad_if_not_divisible=True)[0]
        return f.keypoints.float().cpu().numpy(), f.descriptors.float().cpu().numpy()

    @torch.no_grad()
    def match(self, kp1, d1, kp2, d2, size1, size2):
        """size = (width, height). Returns idx1, idx2, stats."""
        dev = device()
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()[None].to(dev)
        data = dict(
            image0=dict(keypoints=t(kp1), descriptors=t(d1), image_size=t(np.array(size1))),
            image1=dict(keypoints=t(kp2), descriptors=t(d2), image_size=t(np.array(size2))),
        )
        out = self.lg(data)
        m = out["matches"][0].cpu().numpy()
        return m[:, 0].astype(int), m[:, 1].astype(int), dict(n_raw=len(kp1), n_ratio=len(m), n_mutual=len(m))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_matching(d1, d2, reps=3):
    """Time CPU brute force vs GPU for one descriptor pair (seconds per pair)."""
    import cv2
    from . import matching as M
    t0 = time.time()
    for _ in range(reps):
        M.match_pair(d1, d2, cv2.NORM_L2, 0.8, True)
    cpu = (time.time() - t0) / reps
    match_pair_gpu(d1, d2); sync()                            # warm-up (CUDA init / kernels)
    t0 = time.time()
    for _ in range(reps):
        match_pair_gpu(d1, d2); sync()
    gpu = (time.time() - t0) / reps
    return cpu, gpu
