"""Visualisation helpers: matches, point cloud + cameras, reprojection overlays, PLY export, turntable video."""
from __future__ import annotations

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import datasets as D
from . import geometry as G


# ------------------------------------------------------------------------------- matches
def draw_matches(img1, img2, p1, p2, colors=None, max_draw=250, seed=0, title=None, thickness=1):
    """Side-by-side match drawing. colors: list of BGR per match (default green)."""
    h = max(img1.shape[0], img2.shape[0])
    canvas = np.zeros((h, img1.shape[1] + img2.shape[1], 3), np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1]:] = img2
    n = len(p1)
    idx = np.arange(n)
    if n > max_draw:
        idx = np.random.default_rng(seed).choice(n, max_draw, replace=False)
    off = img1.shape[1]
    for k in idx:
        c = (0, 220, 0) if colors is None else colors[k]
        a = tuple(np.round(p1[k]).astype(int))
        b = (int(round(p2[k][0])) + off, int(round(p2[k][1])))
        cv2.line(canvas, a, b, c, thickness, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 3, c, -1, cv2.LINE_AA)
    if title:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_match_stages(rec, i, j, out_dir, max_draw=200):
    """Figure set for the report: raw NN -> ratio/mutual -> RANSAC (correct=green / incorrect=red via GT)."""
    from . import matching as Mt
    from . import evaluate as Ev
    os.makedirs(out_dir, exist_ok=True)
    img1, _ = D.load_image(rec.ds, rec.names[i], rec.cfg.max_dim)
    img2, _ = D.load_image(rec.ds, rec.names[j], rec.cfg.max_dim)
    norm = __import__("sfm.features", fromlist=["x"]).norm_for(rec.cfg.detector)
    r1, r2 = Mt.raw_nn(rec.descs[i], rec.descs[j], norm)
    a1, a2, st = Mt.match_pair(rec.descs[i], rec.descs[j], norm, rec.cfg.ratio, rec.cfg.cross_check)
    v1, v2 = rec.pair_matches(i, j)
    tag = f"{rec.names[i][:-4]}_{rec.names[j][:-4]}"
    res = {}
    for stage, (x1, x2) in dict(a_raw=(r1, r2), b_ratio_mutual=(a1, a2), c_ransac=(v1, v2)).items():
        col = None
        n_bad = None
        ok = Ev.match_correctness(rec, i, j, x1, x2)
        if ok is not None:
            col = [(0, 220, 0) if o else (0, 0, 255) for o in ok]
            n_bad = int((~ok).sum())
        label = dict(a_raw="1) raw nearest-neighbour", b_ratio_mutual="2) ratio test + mutual check",
                     c_ransac="3) after RANSAC (essential matrix)")[stage]
        title = f"{label}: {len(x1)} matches" + (f" ({n_bad} wrong vs GT epipolar geometry)" if n_bad is not None else "")
        canvas = draw_matches(img1, img2, rec.kps[i][x1], rec.kps[j][x2], col, max_draw, title=title)
        cv2.imwrite(os.path.join(out_dir, f"matches_{tag}_{stage}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        res[stage] = dict(n=int(len(x1)), n_wrong=n_bad)
    return res


# ------------------------------------------------------------------------------- 3D
def _frustum_lines(R, t, K, size, w, h):
    C = G.camera_center(R, t)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = np.hstack([(corners - K[:2, 2]) / np.array([K[0, 0], K[1, 1]]), np.ones((4, 1))])
    pts = C + size * (rays @ R)      # R^T applied to row vectors
    segs = [[C, p] for p in pts] + [[pts[k], pts[(k + 1) % 4]] for k in range(4)]
    return segs, C, R.T @ np.array([0, 0, 1.0])


def clean_mask(xyz, k=4.0):
    """Boolean mask dropping far-away low-parallax outliers: keep points within k x median distance of the median."""
    if len(xyz) < 10:
        return np.ones(len(xyz), bool)
    d = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
    return d < k * np.median(d)


def robust_bounds(xyz, lo=1, hi=99, pad=0.15):
    a, b = np.percentile(xyz, lo, axis=0), np.percentile(xyz, hi, axis=0)
    c, r = (a + b) / 2, (b - a).max() / 2 * (1 + pad)
    return c, r


def _scene_bounds(xyz, Cs):
    """Axis bounds from the cleaned cloud plus cameras (so outliers never squash the picture)."""
    m = clean_mask(xyz)
    return robust_bounds(np.vstack([xyz[m], Cs]), lo=0, hi=100, pad=0.08), m


def plot_scene(rec, pc, path, views=((25, -60), (90, -90), (0, -90)), gt_overlay=False, point_size=1.2, title=None):
    """Point cloud + camera frusta + viewing directions from several viewpoints (report figure)."""
    xyz, rgb = pc["xyz"], pc["rgb"] / 255.0
    Cs = np.array([G.camera_center(*rec.cams[i]) for i in rec.cams])
    (c, r), keep = _scene_bounds(xyz, Cs)
    w, h = rec.K[0, 2] * 2, rec.K[1, 2] * 2
    fs = r * 0.07
    fig = plt.figure(figsize=(6 * len(views), 6))
    for v, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), v + 1, projection="3d")
        ax.scatter(*xyz[keep].T, c=rgb[keep], s=point_size, depthshade=False, linewidths=0)
        for i in rec.cams:
            segs, C, d = _frustum_lines(*rec.cams[i], rec.K, fs, w, h)
            for s in segs:
                ax.plot(*zip(*s), color="tab:red", lw=0.8)
            ax.quiver(*C, *(d * fs * 1.6), color="tab:blue", lw=1.0, arrow_length_ratio=0.25)
        ax.plot(Cs[:, 0], Cs[:, 1], Cs[:, 2], color="k", lw=0.6, ls=":")
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=9)
    fig.suptitle(title or f"{int(keep.sum())} of {len(xyz)} points shown (far outliers hidden), {len(rec.cams)} cameras "
                          f"(red=frustum, blue=viewing direction)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_ply(path, xyz, rgb=None):
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    with open(path, "wb") as f:
        hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        if rgb is not None:
            hdr += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f.write((hdr + "end_header\n").encode())
        if rgb is None:
            f.write(xyz.tobytes())
        else:
            rec = np.empty(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            rec["p"], rec["c"] = xyz, np.asarray(rgb, np.uint8)
            f.write(rec.tobytes())


def write_cameras_ply(path, rec, scale):
    """Camera frusta as a coloured line-set PLY (open together with the cloud in MeshLab/CloudCompare)."""
    w, h = rec.K[0, 2] * 2, rec.K[1, 2] * 2
    pts = []
    for i in rec.cams:
        segs, C, d = _frustum_lines(*rec.cams[i], rec.K, scale, w, h)
        for a, b in segs:
            for s in np.linspace(0, 1, 25):
                pts.append(a + s * (b - a))
    pts = np.array(pts)
    write_ply(path, pts, np.tile(np.array([[255, 0, 0]], np.uint8), (len(pts), 1)))


# ------------------------------------------------------------------------------- reprojection
def reprojection_figure(rec, im, path, max_pts=400, seed=0):
    """Detected features (green) vs reprojected 3D points (red) joined by error lines, plus error histogram."""
    img, _ = D.load_image(rec.ds, rec.names[im], rec.cfg.max_dim)
    R, t = rec.cams[im]
    pids, det, rep, err = [], [], [], []
    for pid, tr in rec.tracks.items():
        if im in tr:
            x = rec.kps[im][tr[im]]
            uv, z = G.project(rec.K, R, t, rec.points[pid][None])
            pids.append(pid); det.append(x); rep.append(uv[0]); err.append(np.linalg.norm(uv[0] - x))
    det, rep, err = np.array(det), np.array(rep), np.array(err)
    if len(det) == 0:
        return None
    sel = np.arange(len(det))
    if len(sel) > max_pts:
        sel = np.random.default_rng(seed).choice(len(det), max_pts, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[2.1, 1]))
    ax[0].imshow(img[:, :, ::-1])
    for k in sel:
        ax[0].plot([det[k, 0], rep[k, 0]], [det[k, 1], rep[k, 1]], color="yellow", lw=0.8)
    ax[0].scatter(*det[sel].T, s=10, facecolors="none", edgecolors="lime", label="detected feature")
    ax[0].scatter(*rep[sel].T, s=6, c="red", label="reprojected 3D point")
    ax[0].legend(loc="lower right"); ax[0].axis("off")
    ax[0].set_title(f"{rec.names[im]}: {len(det)} observations, mean err {err.mean():.2f}px "
                    f"(error lines magnified x1)")
    ax[1].hist(err, bins=40, color="tab:blue")
    ax[1].axvline(np.median(err), color="k", ls="--", label=f"median {np.median(err):.2f}px")
    ax[1].set_xlabel("reprojection error (px)"); ax[1].set_ylabel("count"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return dict(n=len(det), mean=float(err.mean()), median=float(np.median(err)))


def error_histogram(rec, path):
    e = rec.observation_errors()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(e, bins=60, color="tab:blue")
    ax.axvline(np.median(e), color="k", ls="--", label=f"median {np.median(e):.2f}px")
    ax.axvline(e.mean(), color="tab:red", ls="--", label=f"mean {e.mean():.2f}px")
    ax.set_xlabel("reprojection error (px)"); ax.set_ylabel("observations"); ax.legend()
    ax.set_title("Reprojection error over all observations")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------- turntable video
def turntable_video(rec, pc, path, frames=180, size=(960, 720), elev=18, fps=30, point_size=1.5):
    """Render a rotating view of cloud + cameras to an mp4 (for the demonstration video)."""
    xyz, rgb = pc["xyz"], pc["rgb"] / 255.0
    Cs = np.array([G.camera_center(*rec.cams[i]) for i in rec.cams])
    (c, r), keep = _scene_bounds(xyz, Cs)
    w, h = rec.K[0, 2] * 2, rec.K[1, 2] * 2
    fs = r * 0.06
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for f in range(frames):
        ax.clear()
        ax.scatter(*xyz[keep].T, c=rgb[keep], s=point_size, depthshade=False, linewidths=0)
        for i in rec.cams:
            segs, C, d = _frustum_lines(*rec.cams[i], rec.K, fs, w, h)
            for s in segs:
                ax.plot(*zip(*s), color="tab:red", lw=0.7)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=elev, azim=-60 + 360 * f / frames)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1]
        vw.write(cv2.resize(np.ascontiguousarray(frame), size))
    vw.release()
    plt.close(fig)
