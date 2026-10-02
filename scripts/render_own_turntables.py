"""Object-focused turntable videos from the dense object clouds (fast numpy point renderer, orthographic, depth-sorted).

    python scripts/render_own_turntables.py
Writes results/own/<tag>/turntable_object.mp4 (960x720, 30 fps, 150 frames = one full turn, slight downward tilt).
"""
import os, pickle, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import viz

JOBS = [("yellow_all", "dense_object.ply"), ("fire", "dense_object_v2.ply"), ("snacks", "dense_object.ply"),
        ("bottle_classic", "dense_object.ply"), ("bottle_learned", "dense_object.ply")]
W, H, FRAMES, FPS = 960, 720, 150, 30


def read_ply(path):
    raw = open(path, "rb").read()
    k = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(raw[:k].split(b"element vertex ")[1].split()[0])
    r = np.frombuffer(raw[k:k + 15 * n], dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    return r["p"].astype(np.float64), r["c"].copy()


def render(P, C, theta, tilt, scale, center, splat=2):
    """Rotate about the vertical (z) axis by theta, tilt the camera down by `tilt`, orthographic projection, far-to-near."""
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    ct, st = np.cos(tilt), np.sin(tilt)
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    Q = (P - center) @ Rz.T @ Rx.T                       # x right, y depth (away), z up
    u = (W / 2 + Q[:, 0] * scale).astype(int)
    v = (H / 2 - Q[:, 2] * scale).astype(int)
    order = np.argsort(-Q[:, 1])                          # far first, near last -> near points overwrite
    img = np.full((H, W, 3), 245, np.uint8)
    for dx in range(splat):
        for dy in range(splat):
            uu, vv = u[order] + dx, v[order] + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[ok], uu[ok]] = C[order][ok][:, ::-1]
    return img


for tag, ply in JOBS:
    rec = pickle.load(open(f"results/own/{tag}/rec.pkl", "rb"))
    M = viz.level_matrix(rec)                             # rows: right, forward, up (world coords)
    P, C = read_ply(f"results/own/{tag}/{ply}")
    P = P[viz.clean_mask(P, 4.0)]; C = C[viz.clean_mask(read_ply(f"results/own/{tag}/{ply}")[0], 4.0)]
    Q = P @ np.vstack([M[0], M[1], M[2]]).T               # leveled: x right, y forward, z up
    center = np.array([np.median(Q[:, 0]), np.median(Q[:, 1]), np.median(Q[:, 2])])
    ext = max(np.percentile(Q[:, 0], 98) - np.percentile(Q[:, 0], 2), np.percentile(Q[:, 2], 98) - np.percentile(Q[:, 2], 2))
    scale = 0.78 * min(W, H) / ext
    if len(Q) > 600000:
        sel = np.random.default_rng(0).choice(len(Q), 600000, replace=False); Q, C = Q[sel], C[sel]
    out = f"results/own/{tag}/turntable_object.mp4"
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(FRAMES):
        vw.write(render(Q, C, 2 * np.pi * f / FRAMES, np.radians(12), scale, center, splat=2 if len(Q) < 250000 else 1))
    vw.release()
    print(f"{tag}: {len(Q):,} points -> {out}", flush=True)
