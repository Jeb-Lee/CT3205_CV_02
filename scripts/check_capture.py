"""Quality check of a freshly captured image set, BEFORE reconstruction.

    python scripts/check_capture.py                      # all five sets, EXIF + sharpness + exposure + contact sheet
    python scripts/check_capture.py --matching           # also neighbour-overlap test (RootSIFT + F-matrix RANSAC)
    python scripts/check_capture.py --sets bottle snacks

Checks (all against what the SfM pipeline needs):
  * camera settings constant over the set (focal length, resolution, orientation tag, exposure mode) - changes break the
    single-camera-matrix model; portrait/EXIF-rotated frames caused our worst silent failure on Person Hall
  * sharpness (variance of the Laplacian, relative to the set median), clipped highlights / crushed shadows
  * capture order gaps (missing file numbers = skipped / deleted shots)
  * neighbour overlap: verified inliers between frame i and i+1 / i+2 -> pair-graph connectivity (the failure mode of the
    whole-loop experiment). A set whose graph is disconnected cannot be reconstructed incrementally.
Writes results/own/qa_<set>.json, contact_<set>.jpg and (with --matching) overlap_<set>.png.
"""
import argparse
import json
import os
import re
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "own")
SETS = {
    "bottle": "Datasets/Bottle",
    "fire_ext": "Datasets/Fire est",
    "snacks": "Datasets/Snacks",
    "yellow_high": "Datasets/yellow_sign/Yellow High",
    "yellow_level": "Datasets/yellow_sign/Yellow_Hori",
}
TAGS = {"Make": 0x010F, "Model": 0x0110, "Orientation": 0x0112}
EXIF_IFD = {"ExposureTime": 0x829A, "FNumber": 0x829D, "ISO": 0x8827, "FocalLength": 0x920A, "Flash": 0x9209,
            "FocalLength35": 0xA405, "DigitalZoom": 0xA404, "ExposureMode": 0xA402, "WhiteBalance": 0xA403,
            "LensModel": 0xA434, "DateTimeOriginal": 0x9003, "SceneType": 0xA301}


def read_exif(path):
    im = Image.open(path)
    ex = im.getexif()
    d = {k: ex.get(v) for k, v in TAGS.items()}
    ifd = ex.get_ifd(0x8769)
    for k, v in EXIF_IFD.items():
        d[k] = ifd.get(v)
    d["size"] = im.size  # as stored (width, height)
    for k in ("ExposureTime", "FNumber", "FocalLength"):
        if d.get(k) is not None:
            d[k] = float(d[k])
    return d


def number(name):
    m = re.search(r"(\d+)", os.path.splitext(name)[0])
    return int(m.group(1)) if m else None


def analyse_set(key, folder, do_matching):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    n = len(names)
    print(f"\n=== {key}: {n} images in {folder}", flush=True)
    ex = [read_exif(os.path.join(folder, f)) for f in names]
    rep = dict(set=key, folder=folder, n_images=n)

    # ---- camera / settings consistency
    def uniq(field):
        vals = [e.get(field) for e in ex]
        return sorted({str(v) for v in vals})
    rep["camera"] = f"{ex[0].get('Make')} {ex[0].get('Model')}"
    rep["stored_sizes"] = uniq("size")
    rep["orientation_tags"] = uniq("Orientation")
    rep["focal_length_mm"] = uniq("FocalLength")
    rep["focal_35mm"] = uniq("FocalLength35")
    rep["lens"] = uniq("LensModel")
    rep["digital_zoom"] = uniq("DigitalZoom")
    rep["exposure_mode"] = uniq("ExposureMode")
    rep["white_balance"] = uniq("WhiteBalance")
    et = np.array([e["ExposureTime"] for e in ex if e.get("ExposureTime")])
    iso = np.array([e["ISO"] for e in ex if e.get("ISO")])
    rep["exposure_time_s"] = dict(min=float(et.min()), median=float(np.median(et)), max=float(et.max())) if len(et) else None
    rep["iso"] = dict(min=int(iso.min()), median=int(np.median(iso)), max=int(iso.max())) if len(iso) else None
    nums = [number(f) for f in names]
    gaps = [(nums[i], nums[i + 1]) for i in range(n - 1) if nums[i + 1] - nums[i] > 1]
    rep["numbering_gaps"] = gaps

    # ---- pixel statistics + sharpness
    lap, mean_l, clip_hi, clip_lo, small, feats = [], [], [], [], [], []
    for f in names:
        im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h, w = im.shape[:2]
        s = 1000.0 / max(h, w)
        g = cv2.cvtColor(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        lap.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        mean_l.append(float(g.mean()))
        clip_hi.append(float((g >= 250).mean())); clip_lo.append(float((g <= 5).mean()))
        small.append(cv2.resize(im, (360, int(360 * h / w)) if w >= h else (int(360 * w / h), 360), interpolation=cv2.INTER_AREA))
    lap = np.array(lap)
    med = float(np.median(lap))
    soft = [names[i] for i in range(n) if lap[i] < 0.5 * med]
    rep["sharpness"] = dict(median_laplacian_var=med, min=float(lap.min()), max=float(lap.max()),
                            soft_images_below_half_median=soft, n_soft=len(soft))
    rep["brightness"] = dict(mean=float(np.mean(mean_l)), min=float(np.min(mean_l)), max=float(np.max(mean_l)),
                             max_clipped_highlights=float(np.max(clip_hi)), max_crushed_shadows=float(np.max(clip_lo)))
    print(f"  camera: {rep['camera']} | sizes {rep['stored_sizes']} | orientation tags {rep['orientation_tags']}")
    print(f"  focal {rep['focal_length_mm']} mm (35mm-eq {rep['focal_35mm']}) | zoom {rep['digital_zoom']} | lens {rep['lens']}")
    print(f"  exposure time {rep['exposure_time_s']} | ISO {rep['iso']}")
    print(f"  numbering gaps (missing shots): {gaps[:8]}{' ...' if len(gaps) > 8 else ''}")
    print(f"  sharpness: median {med:.0f}, min {lap.min():.0f}; soft (<0.5x median): {len(soft)} {soft[:6]}")
    print(f"  brightness mean {np.mean(mean_l):.0f} (range {np.min(mean_l):.0f}-{np.max(mean_l):.0f}); worst clipped highlights {np.max(clip_hi) * 100:.1f}%", flush=True)

    # ---- contact sheet (up to 24 tiles)
    step = max(1, n // 24)
    tiles = []
    for i in range(0, n, step)[:24]:
        t = cv2.resize(small[i], (300, 225))
        cv2.putText(t, f"{i}:{names[i][-8:-4]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(t, f"{i}:{names[i][-8:-4]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros((225, 300, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)])
    cv2.imwrite(os.path.join(OUT, f"contact_{key}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- neighbour overlap
    if do_matching:
        sift = cv2.SIFT_create(nfeatures=4000)
        kps, des = [], []
        for f in names:
            im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            s = 1000.0 / max(im.shape)
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            k, d = sift.detectAndCompute(im, None)
            d = d / (np.abs(d).sum(1, keepdims=True) + 1e-7)
            kps.append(np.array([p.pt for p in k], np.float32)); des.append(np.sqrt(d).astype(np.float32))
        rep["features_per_image"] = dict(median=int(np.median([len(k) for k in kps])), min=int(min(len(k) for k in kps)))
        bf = cv2.BFMatcher(cv2.NORM_L2)

        def inliers(i, j):
            m = bf.knnMatch(des[i], des[j], k=2)
            good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
            if len(good) < 12:
                return len(good), 0
            p1 = kps[i][[g.queryIdx for g in good]]; p2 = kps[j][[g.trainIdx for g in good]]
            _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, 0.999)
            return len(good), int(mask.sum()) if mask is not None else 0

        g1 = [inliers(i, i + 1)[1] for i in range(n - 1)]
        g2 = [inliers(i, i + 2)[1] for i in range(n - 2)]
        # connectivity of the pair graph (edge = >= 40 inliers within +-3 frames)
        adj = np.zeros((n, n), bool)
        for i in range(n):
            for j in range(i + 1, min(n, i + 4)):
                if j == i + 1: v = g1[i]
                elif j == i + 2: v = g2[i]
                else: v = inliers(i, j)[1]
                adj[i, j] = adj[j, i] = v >= 40
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        k_, lab = connected_components(csr_matrix(adj), directed=False)
        weak = [(i, names[i][-8:-4], g1[i]) for i in range(n - 1) if g1[i] < 60]
        rep["overlap"] = dict(inliers_gap1_median=int(np.median(g1)), inliers_gap1_min=int(min(g1)),
                              inliers_gap2_median=int(np.median(g2)), inliers_gap2_min=int(min(g2)),
                              weak_neighbour_links_lt60=weak, graph_components=int(k_), largest_component=int(np.bincount(lab).max()))
        print(f"  neighbour inliers: gap1 median {rep['overlap']['inliers_gap1_median']} (min {min(g1)}), gap2 median {rep['overlap']['inliers_gap2_median']} (min {min(g2)})")
        print(f"  pair graph: {k_} component(s), largest {rep['overlap']['largest_component']}/{n}; weak neighbour links (<60 inliers): {len(weak)} {weak[:6]}", flush=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].plot(g1, label="frame i -> i+1"); ax[0].plot(g2, label="i -> i+2"); ax[0].axhline(40, color="r", ls="--", lw=.8)
        ax[0].set_yscale("log"); ax[0].set_xlabel("frame index"); ax[0].set_ylabel("RANSAC inliers"); ax[0].legend(); ax[0].set_title(f"{key}: overlap with next frames")
        ax[1].plot(lap, color="tab:green"); ax[1].axhline(0.5 * med, color="r", ls="--", lw=.8); ax[1].set_xlabel("frame index"); ax[1].set_ylabel("sharpness (Laplacian var)")
        ax[1].set_title("sharpness (red = 0.5 x median)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"overlap_{key}.png"), dpi=110); plt.close(fig)
    json.dump(rep, open(os.path.join(OUT, f"qa_{key}.json"), "w"), indent=1, default=str)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--matching", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for k in a.sets:
        analyse_set(k, os.path.join(ROOT, SETS[k]), a.matching)
