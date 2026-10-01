"""Our reconstructions vs the COLMAP model of the dataset, on exactly the same images (advanced work: COLMAP comparison).

    python scripts/run_colmap_compare.py
Writes results/colmap_compare/{comparison.csv, comparison.md, bars.png, overlay_<case>.png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, viz
from sfm.colmap_compare import compare
from sfm.reconstruction import Reconstruction, SfMConfig

OUT = os.path.join("results", "colmap_compare")
os.makedirs(OUT, exist_ok=True)

CASES = [  # label, dataset, start, count, stride, cfg overrides
    ("south 12 imgs (stride 2)", "south-building", 40, 12, 2, {}),
    ("south 12 imgs, DISK+LightGlue", "south-building", 40, 12, 2, dict(detector="disk", matcher="lightglue")),
    ("south 16 consecutive", "south-building", 40, 16, 1, {}),
    ("hall A clean facade", "person-hall", 105, 12, 3, {}),
    ("hall C arches + flare", "person-hall", 205, 12, 2, {}),
]


def overlay(rec, pc, keep, sim3, ds, path, title):
    """Our cloud (red) and COLMAP's points restricted to the same images (blue), both in COLMAP's frame."""
    s, R, t = sim3
    ours = s * pc["xyz"] @ R.T + t
    col = ds.gt_points[keep]
    m = viz.clean_mask(ours, 3.0)
    c, r = viz.robust_bounds(np.vstack([ours[m], col]), lo=1, hi=99, pad=0.05)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # three orthographic projections (COLMAP frame is y-down like the camera frame)
    for ax, (a, b, na, nb) in zip(axs, ((0, 1, "x", "y"), (0, 2, "x", "z"), (2, 1, "z", "y"))):
        ax.scatter(col[:, a], col[:, b], s=0.6, c="tab:blue", alpha=0.5, label=f"COLMAP ({len(col):,} pts)")
        ax.scatter(ours[m][:, a], ours[m][:, b], s=0.6, c="tab:red", alpha=0.5, label=f"ours ({int(m.sum()):,} pts)")
        ax.set_xlim(c[a] - r, c[a] + r); ax.set_ylim(c[b] - r, c[b] + r); ax.set_aspect("equal")
        if b == 1:
            ax.invert_yaxis()
        ax.set_xlabel(na); ax.set_ylabel(nb); ax.legend(markerscale=8, fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    cache = {}
    rows = []
    for label, dname, start, count, stride, over in CASES:
        ds = cache.setdefault(dname, D.load_dataset(dname))
        names = D.select_images(ds, start, count, stride)
        print(f"\n=== {label}", flush=True)
        rec = Reconstruction(ds, names, SfMConfig(**over), cache_dir="results/cache", verbose=False).run()
        pc = rec.point_cloud(with_colors=False)
        r = compare(rec, pc, include_mask=True)
        keep, sim3 = r.pop("_keep"), r.pop("_sim3")
        r.update(label=label, seconds=rec.log["timing"].get("total", 0.0))
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if sim3 is not None and label in ("south 12 imgs (stride 2)", "hall A clean facade"):
            overlay(rec, pc, keep, sim3, ds, os.path.join(OUT, f"overlay_{label.split()[0]}_{label.split()[1]}.png"),
                    f"{label}: our sparse cloud vs COLMAP points seen by the same images (Sim(3)-aligned)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    cols = ["label", "n_images", "ours_points", "colmap_points", "point_ratio_ours_over_colmap", "ours_mean_track_len",
            "colmap_mean_track_len", "ours_points_per_image", "colmap_points_per_image", "ours_reproj_px", "colmap_reproj_px",
            "cam_pos_rmse_rel", "cam_rot_mean_deg", "seconds"]
    try:
        open(os.path.join(OUT, "comparison.md"), "w").write(df[cols].to_markdown(index=False, floatfmt=".4g"))
    except ImportError:
        open(os.path.join(OUT, "comparison.md"), "w").write(df[cols].to_string())
    print("\n", df[cols].round(4).to_string(index=False))

    fig, axs = plt.subplots(1, 4, figsize=(19, 4.4))
    x = np.arange(len(df)); w = 0.38
    for ax, (a, b, ttl) in zip(axs, (("ours_points", "colmap_points", "3D points (same images)"),
                                      ("ours_mean_track_len", "colmap_mean_track_len", "mean track length"),
                                      ("ours_points_per_image", "colmap_points_per_image", "observations per image"),
                                      ("ours_reproj_px", "colmap_reproj_px", "mean reprojection error (original-res px)"))):
        ax.bar(x - w / 2, df[a], w, color="tab:red", label="ours"); ax.bar(x + w / 2, df[b], w, color="tab:blue", label="COLMAP")
        ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n", 2) for l in df.label], fontsize=6); ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=.3, axis="y")
    axs[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "bars.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
