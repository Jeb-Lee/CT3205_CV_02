"""Compose the demonstration video clips from a reconstruction (run BEFORE screen-recording the live viewer).

    python scripts/make_demo_video.py --start 40 --count 12 --stride 2 --out results/demo

Produces results/demo/demo_auto.mp4 with, in order:
  1. title card + input images strip
  2. feature-match stages (raw -> ratio/mutual -> RANSAC) for consecutive pairs
  3. cameras appearing one by one in registration order with the growing point cloud
  4. 360-degree turntable of the final cloud + cameras
  5. reprojection overlays
Add your own voice-over / live Open3D segment in any video editor (or OBS), see docs/GUIDE_YOUR_TODO.md.
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, viz, geometry as G, matching as Mt, features as F
from sfm.reconstruction import Reconstruction, SfMConfig

W, H, FPS = 1280, 720, 24


def card(text, sub=""):
    im = np.full((H, W, 3), 25, np.uint8)
    cv2.putText(im, text, (60, H // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(im, sub, (60, H // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return im


def fit(im):
    h, w = im.shape[:2]
    s = min(W / w, H / h)
    im = cv2.resize(im, (int(w * s), int(h * s)))
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - im.shape[0]) // 2, (W - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


def hold(vw, im, sec):
    for _ in range(int(sec * FPS)):
        vw.write(im)


def label(im, txt):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (W, 40), (0, 0, 0), -1)
    cv2.putText(im, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def grow_frames(rec, pc, vw, per_cam=int(1.0 * FPS)):
    """Show the cloud as cameras are registered one by one (points restricted to those visible so far)."""
    order = [e for e in rec.log["events"] if e["type"] in ("init", "register")]
    reg_order = list(rec.init_pair) + [rec.names.index(e["image"]) for e in order if e["type"] == "register"]
    M = viz.level_matrix(rec)
    Cs = np.array([G.camera_center(*rec.cams[i]) for i in rec.cams]) @ M.T
    pc = dict(pc, xyz=pc["xyz"] @ M.T)
    (c, r), keep = viz._scene_bounds(pc["xyz"], Cs)
    w, h = rec.K[0, 2] * 2, rec.K[1, 2] * 2
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([-0.04, -0.02, 1.08, 1.10])         # moderate zoom: the camera arc must stay in frame
    pids = list(pc["pids"])
    for n in range(2, len(reg_order) + 1):
        active = set(reg_order[:n])
        sel = np.array([keep[k] and any(im in active for im in rec.tracks[p]) for k, p in enumerate(pids)])
        ax.clear()
        ax.scatter(*pc["xyz"][sel].T, c=pc["rgb"][sel] / 255.0, s=4.0, depthshade=False, linewidths=0)
        for i in reg_order[:n]:
            segs, C, d = viz._frustum_leveled(rec, i, r * 0.09, w, h, M)
            for s in segs:
                ax.plot(*zip(*s), color="tab:red", lw=1.2)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=-68); ax.set_axis_off()
        fig.suptitle(f"{n} cameras registered - {int(sel.sum()):,} points", fontsize=22, y=0.95)
        fig.canvas.draw()
        fr = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1])
        hold(vw, cv2.resize(fr, (W, H)), per_cam / FPS)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="south-building")
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="results/demo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = D.load_dataset(a.dataset)
    names = D.select_images(ds, a.start, a.count, a.stride)
    rec = Reconstruction(ds, names, SfMConfig(), cache_dir="results/cache").run()
    pc = rec.point_cloud()
    vw = cv2.VideoWriter(os.path.join(a.out, "demo_auto.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    hold(vw, card("Multi-view 3D reconstruction", f"{a.dataset}: {len(names)} images -> sparse point cloud"), 3)
    # 1. input images
    for n in names:
        im, _ = D.load_image(ds, n, 1280)
        hold(vw, label(fit(im), f"input image {n}"), 0.6)
    # 2. match stages for consecutive pairs of the sequence
    norm = F.norm_for(rec.cfg.detector)
    for (i, j) in [(0, 1), (3, 4), (6, 7)]:
        img1, _ = D.load_image(ds, names[i], rec.cfg.max_dim)
        img2, _ = D.load_image(ds, names[j], rec.cfg.max_dim)
        r1, r2 = Mt.raw_nn(rec.descs[i], rec.descs[j], norm)
        a1, a2, _ = Mt.match_pair(rec.descs[i], rec.descs[j], norm, rec.cfg.ratio, rec.cfg.cross_check)
        v1, v2 = rec.pair_matches(i, j)
        for txt, (x1, x2) in [("raw nearest-neighbour matches", (r1, r2)), ("after ratio test + mutual check", (a1, a2)),
                              ("after RANSAC on the essential matrix", (v1, v2))]:
            im = viz.draw_matches(img1, img2, rec.kps[i][x1], rec.kps[j][x2], None, 150, thickness=1)
            hold(vw, label(fit(im), f"{names[i]} <-> {names[j]}: {txt} ({len(x1)})"), 2.0)
    # 3. incremental growth
    hold(vw, card("Incremental reconstruction", "cameras are added by PnP; new points by triangulation; BA refines all"), 2)
    grow_frames(rec, pc, vw)
    # 4. turntable
    tmp = os.path.join(a.out, "_turn.mp4")
    viz.turntable_video(rec, pc, tmp, frames=150, size=(W, H), fps=FPS, elev=12, point_size=2.5)
    cap = cv2.VideoCapture(tmp)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
    cap.release(); os.remove(tmp)
    # 5. reprojection
    regs = sorted(rec.cams)
    for im in [regs[0], regs[len(regs) // 2], regs[-1]]:
        p = os.path.join(a.out, "_rp.jpg")
        viz.reprojection_figure(rec, im, p)
        hold(vw, label(fit(cv2.imread(p)), "reprojection: detected (green) vs reprojected (red)"), 3.0)
    os.remove(p)
    hold(vw, card("Done", f"{len(rec.points)} points, {len(rec.cams)} cameras"), 2)
    vw.release()
    print("wrote", os.path.join(a.out, "demo_auto.mp4"))


if __name__ == "__main__":
    main()
