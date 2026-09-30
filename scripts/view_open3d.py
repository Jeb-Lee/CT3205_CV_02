"""Interactive 3D viewer: point cloud + camera frusta (+ optional COLMAP ground truth overlay).

    python scripts/view_open3d.py results/baseline            # our reconstruction
    python scripts/view_open3d.py results/baseline --gt       # + COLMAP cloud in grey (after Sim3 alignment)

Mouse: left-drag rotate, wheel zoom, ctrl+left-drag pan.  Press 'S' is NOT bound - use Win+Shift+S for screenshots
(or --shot out.png to save an offscreen-free capture from the window).
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_ply(path):
    return o3d.io.read_point_cloud(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="results/baseline")
    ap.add_argument("--gt", action="store_true", help="overlay COLMAP ground-truth points, aligned to ours")
    ap.add_argument("--dataset", default="south-building")
    ap.add_argument("--point_size", type=float, default=2.5)
    ap.add_argument("--shot", default=None, help="save a screenshot to this path and exit")
    a = ap.parse_args()

    geoms = []
    cloud = load_ply(os.path.join(a.folder, "cloud.ply"))
    xyz = np.asarray(cloud.points)
    # drop far outliers for a usable initial view
    d = np.linalg.norm(xyz - np.median(xyz, 0), axis=1)
    keep = np.flatnonzero(d < 4 * np.median(d))
    cloud = cloud.select_by_index(keep)
    geoms.append(cloud)

    cams = load_ply(os.path.join(a.folder, "cameras.ply"))
    geoms.append(cams)

    if a.gt:
        from sfm import datasets as D, evaluate as Ev, geometry as G
        import json
        ds = D.load_dataset(a.dataset)
        st = json.load(open(os.path.join(a.folder, "stats.json")))
        # align COLMAP -> our frame using camera centres of registered images
        from sfm.reconstruction import Reconstruction  # noqa: F401 (kept for symmetry)
        # our camera centres are not stored separately, so recover the transform from the .ply of frusta apexes is
        # brittle; instead stats.json holds the alignment scale/R/t if the baseline script saved it.
        if "sim3" in st:
            s, R, t = st["sim3"]["s"], np.array(st["sim3"]["R"]), np.array(st["sim3"]["t"])
            # ours -> gt is X_gt = s R X + t  =>  X = R^T (X_gt - t) / s
            gt = (ds.gt_points - t) @ R / s
            g = o3d.geometry.PointCloud()
            g.points = o3d.utility.Vector3dVector(gt)
            g.paint_uniform_color([0.55, 0.55, 0.55])
            geoms.insert(0, g)
        else:
            print("stats.json has no 'sim3'; re-run scripts/run_baseline.py to store it")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Multi-view reconstruction", 1400, 900)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = a.point_size
    opt.background_color = np.array([1, 1, 1])
    if a.shot:
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(a.shot, do_render=True)
        vis.destroy_window()
        return
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
