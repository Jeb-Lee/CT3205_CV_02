"""Challenging-scenario study on the Person Hall dataset: one test case per difficulty, everything logged.

    python scripts/run_person_hall.py --cases A B C D        # run some cases
    python scripts/run_person_hall.py --cases all

Per case -> results/person_hall/<case>/ : stats.json, scene.png, inlier_matrix.png, matches_*.jpg (raw/filtered/RANSAC with
GT-based correct/incorrect colouring), reproj_*.jpg, optional turntable.mp4; and one row in results/person_hall/cases.csv.
Outcome labels (fixed in advance, see classify()): SUCCESS / PARTIAL / FAILURE.
"""
import argparse
import json
import os
import sys
import time
import traceback

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, viz, evaluate as Ev, geometry as G
from sfm.reconstruction import Reconstruction, SfMConfig

OUT = os.path.join("results", "person_hall")
CACHE = os.path.join("results", "cache")

# name -> (title, difficulty it tests, start, count, stride, cfg overrides)
CASES = {
    "A": ("clean facade + obelisk", "reference case: near-planar brick facade, moderate viewpoint change", 105, 12, 3, {}),
    "B": ("foliage occlusion", "trees and shrubs cover most of the building; unstable, view-dependent texture", 30, 12, 3, {}),
    "C": ("repetitive arched windows + sun flare", "identical arched windows / blind arches; lens flare in frame 210", 205, 12, 2, {}),
    "D": ("corner turn, wide baseline", "camera rounds a building corner; large viewpoint change per step", 270, 12, 4, {}),
    "E2": ("stride 2 on clean facade", "viewpoint sweep, 8 images", 105, 8, 2, {}),
    "E4": ("stride 4 on clean facade", "viewpoint sweep, 8 images", 105, 8, 4, {}),
    "E6": ("stride 6 on clean facade", "viewpoint sweep, 8 images", 105, 8, 6, {}),
    "E8": ("stride 8 on clean facade", "viewpoint sweep, 8 images", 105, 8, 8, {}),
    "F3": ("whole building loop, denser sampling", "every 3rd image around the full loop (110 images), windowed matching", 0, 110, 3,
           dict(window=6, ba_every=10)),
    "F3r": ("whole loop, denser sampling + relaxed thresholds", "as F3 but ratio 0.85, pair-use >= 15 inliers, PnP >= 12", 0, 110, 3,
            dict(window=6, ba_every=10, ratio=0.85, min_pair_use=15, min_pnp=12)),
    # ---- learned features: DISK keypoints + LightGlue matcher (GPU), everything else identical
    "B_dl": ("foliage occlusion, DISK+LightGlue", "case B with learned features", 30, 12, 3, dict(detector="disk", matcher="lightglue")),
    "D_dl": ("corner turn, DISK+LightGlue", "case D with learned features", 270, 12, 4, dict(detector="disk", matcher="lightglue")),
    "F_dl": ("whole loop every 6th, DISK+LightGlue", "case F with learned features", 0, 55, 6,
             dict(detector="disk", matcher="lightglue", window=8, ba_every=6)),
    "F3_dl": ("whole loop every 3rd, DISK+LightGlue", "case F3 with learned features", 0, 110, 3,
              dict(detector="disk", matcher="lightglue", window=6, ba_every=10)),
    # ---- learned + strict pair thresholds (150 verified inliers): learned matches give true pairs thousands of inliers
    "B_dls": ("foliage, DISK+LG strict", "B, learned, min pair 150", 30, 12, 3,
              dict(detector="disk", matcher="lightglue", min_pair_inliers=150, min_pair_use=150)),
    "D_dls": ("corner turn, DISK+LG strict", "D, learned, min pair 150", 270, 12, 4,
              dict(detector="disk", matcher="lightglue", min_pair_inliers=150, min_pair_use=150)),
    "F_dls": ("whole loop every 6th, DISK+LG strict", "F, learned, min pair 150", 0, 55, 6,
              dict(detector="disk", matcher="lightglue", window=8, ba_every=6, min_pair_inliers=150, min_pair_use=150)),
    "F3_dls": ("whole loop every 3rd, DISK+LG strict", "F3, learned, min pair 150", 0, 110, 3,
               dict(detector="disk", matcher="lightglue", window=6, ba_every=10, min_pair_inliers=150, min_pair_use=150)),
    "F": ("whole building loop, sparse sampling", "every 6th image around the full loop (~55 images), windowed matching", 0, 55, 6,
          dict(window=8, ba_every=6)),
}


def classify(st):
    """Fixed outcome rule (decided before looking at results):
    SUCCESS  : >= 90 % of images registered AND camera position error < 2 % of trajectory extent (or no GT available)
    PARTIAL  : >= 50 % registered
    FAILURE  : otherwise (including crashes such as 'no valid initial pair')."""
    if st.get("failed"):
        return "FAILURE"
    frac = st["n_registered"] / st["n_images"]
    err = st.get("cam_pos_rmse_rel", np.nan)
    if frac >= 0.9 and (np.isnan(err) or err < 0.02):
        return "SUCCESS"
    if frac >= 0.5:
        return "PARTIAL"
    return "FAILURE"


def inlier_heatmap(rec, path, title):
    n = rec.n
    M = np.zeros((n, n))
    for (i, j), m in rec.matches.items():
        M[i, j] = M[j, i] = m["n_inl"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(np.log10(M + 1), cmap="viridis")
    ax.set_title(f"verified inliers per image pair (log10)\n{title}", fontsize=10)
    ax.set_xlabel("image index"); ax.set_ylabel("image index")
    for i in range(n):                                    # mark unregistered images
        if i not in rec.cams:
            ax.axhline(i, color="red", lw=0.6, alpha=.7); ax.axvline(i, color="red", lw=0.6, alpha=.7)
    fig.colorbar(im, ax=ax, label="log10(inliers+1)"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def graph_stats(rec):
    """Connectivity of the pair graph (edge = verified pair with enough inliers): a disconnected graph cannot be
    reconstructed by incremental SfM, whatever else is done."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = rec.n
    A = np.zeros((n, n), bool)
    for (i, j), m in rec.matches.items():
        if m["n_inl"] >= rec.cfg.min_pair_use:
            A[i, j] = A[j, i] = True
    k, lab = connected_components(csr_matrix(A), directed=False)
    sizes = np.bincount(lab)
    return dict(n_components=int(k), largest_component=int(sizes.max()), n_singletons=int((sizes == 1).sum()),
                registered_in_largest=int(sum(1 for i in rec.cams if lab[i] == sizes.argmax())))


def run_case(ds, key, video=False):
    title, why, start, count, stride, over = CASES[key]
    out = os.path.join(OUT, key)
    os.makedirs(out, exist_ok=True)
    names = D.select_images(ds, start, count, stride)
    cfg = SfMConfig(**over)
    row = dict(case=key, title=title, tests=why, first=names[0], last=names[-1], n_images=len(names), stride=stride,
               window=cfg.window)
    print(f"\n=== case {key}: {title} | {len(names)} images {names[0]}..{names[-1]} stride {stride}", flush=True)
    t0 = time.time()
    rec = None
    try:
        rec = Reconstruction(ds, names, cfg, cache_dir=CACHE, verbose=True).run()
        pc = rec.point_cloud()
        st = rec.stats()
        gt = Ev.evaluate_against_gt(rec, pc)
        gt.pop("_sim3", None)
        st.update(gt)
        st["failed"] = False
        row.update({k: v for k, v in st.items() if not isinstance(v, (list, dict))})
        json.dump(dict(stats=st, images=names, events=rec.log["events"]), open(os.path.join(out, "stats.json"), "w"),
                  indent=2, default=float)
        if len(rec.cams) >= 2 and len(pc["xyz"]) > 10:
            viz.plot_scene(rec, pc, os.path.join(out, "scene.png"), views=((25, -60), (90, -90)),
                           title=f"case {key} ({title}): {len(pc['xyz'])} pts, {len(rec.cams)}/{len(names)} cameras")
            viz.write_ply(os.path.join(out, "cloud.ply"), pc["xyz"], pc["rgb"])
            regs = sorted(rec.cams)
            viz.reprojection_figure(rec, regs[len(regs) // 2], os.path.join(out, f"reproj_{rec.names[regs[len(regs)//2]][:-4]}.jpg"))
            if video and len(rec.cams) >= 6:
                viz.turntable_video(rec, pc, os.path.join(out, "turntable.mp4"), frames=120, size=(1280, 720), fps=30)
    except Exception as e:                                # recorded as an outcome, never hidden
        traceback.print_exc(limit=3)
        row.update(failed=True, error=f"{type(e).__name__}: {e}", n_registered=0, n_points=0)
    row["seconds_wall"] = time.time() - t0
    row["outcome"] = classify(row)
    if rec is not None and rec.matches:
        try:
            inlier_heatmap(rec, os.path.join(out, "inlier_matrix.png"), f"case {key}: {title}")
            # match-stage figures for the first two images (correct = green, wrong under GT epipolar geometry = red)
            viz.save_match_stages(rec, 0, 1, os.path.join(out, "matches"))
            row.update(graph_stats(rec))
            M = np.zeros((rec.n, rec.n), int)
            for (i, j), m in rec.matches.items():
                M[i, j] = M[j, i] = m["n_inl"]
            np.save(os.path.join(out, "inlier_matrix.npy"), M)
            row["pair01_inliers"] = int(rec.matches[(0, 1)]["n_inl"])
            row["pair01_ratio_mutual"] = int(rec.matches[(0, 1)]["n_mutual"])
        except Exception as e:
            print("  (figures skipped:", e, ")")
        if rec.init_pair is None:                        # failed before init: still dump the matching structure
            row["n_pairs_verified_any"] = int(sum(1 for m in rec.matches.values() if m["n_inl"] >= cfg.min_pair_inliers))
    print(f"--> {key}: {row['outcome']}  reg {row.get('n_registered', 0)}/{len(names)}  pts {row.get('n_points', 0)}  "
          f"camErr {row.get('cam_pos_rmse_rel', float('nan')):.4f}  ({row['seconds_wall']:.0f}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A"])
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    keys = list(CASES) if a.cases == ["all"] else a.cases
    ds = D.load_dataset("person-hall")
    os.makedirs(OUT, exist_ok=True)
    csv = os.path.join(OUT, "cases.csv")
    old = pd.read_csv(csv) if os.path.isfile(csv) else pd.DataFrame()
    rows = []
    for k in keys:
        rows.append(run_case(ds, k, video=a.video))
        new = pd.DataFrame(rows)
        if len(old):
            new = pd.concat([old[~old.case.isin(new.case)], new])
        new.to_csv(csv, index=False)                      # written after every case: a crash never loses results


if __name__ == "__main__":
    main()
