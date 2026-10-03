"""Chessboard calibration of the phone camera (ground truth for our self-calibrated focal length).

    python scripts/make_chessboard.py                                   # creates the printable board (once)
    python scripts/calibrate_chessboard.py --folder Datasets/Calibration --cols 9 --rows 6 --square_mm 25 \
        --skip_dirs measurement misc --no_edited
(cols x rows = number of INNER corners; the printable board has 10 x 7 squares = 9 x 6 inner corners)

* walks sub-folders; skips folders (--skip_dirs), iPhone-EDITED copies IMG_Exxxx (--no_edited: crop/straighten changes the geometry)
  and any photo whose size differs from the majority (a different resolution mode has a different pixel grid)
* detects the board with findChessboardCornersSB, calibrates, drops outlier photos (per-photo RMS > 2.5 x median) and re-calibrates
* compares distortion models (k1 | k1+k2 | k1+k2+tangential) and measures STABILITY with 40 random half-splits of the photos
* reports focal length / principal point at stored resolution AND at the 1600 px working resolution of our pipeline, and the error
  of our self-calibrated value (1154 px at 1600 px)
Writes results/own/chessboard_calibration.json
"""
import argparse, json, os, sys
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square_mm", type=float, default=25.0)
ap.add_argument("--skip_dirs", nargs="*", default=[])
ap.add_argument("--no_edited", action="store_true")
ap.add_argument("--self_calibrated_f_1600", type=float, default=1154.0)
ap.add_argument("--max_dim", type=int, default=1600)
ap.add_argument("--out", default="results/own/chessboard_calibration.json")
a = ap.parse_args()

pattern = (a.cols, a.rows)
objp = np.zeros((a.cols * a.rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_mm


def load_gray(path):
    """JPEG/PNG through OpenCV (stored pixels, orientation tag ignored, like our pipeline); iPhone HEIC/HEIF through pillow-heif."""
    if path.lower().endswith((".heic", ".heif")):
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        return np.array(Image.open(path).convert("L"))
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)


# ------------------------------------------------------------------------------------------ collect + detect
files = []
for dp, dn, fs in os.walk(a.folder):
    dn[:] = [d for d in dn if d not in a.skip_dirs]
    for f in sorted(fs):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif")):
            continue
        if a.no_edited and f.upper().startswith("IMG_E"):
            continue
        files.append(os.path.join(dp, f))
print(f"{len(files)} candidate photos after skipping {a.skip_dirs} {'and edited copies' if a.no_edited else ''}", flush=True)
det = []
for p in files:
    img = load_gray(p)
    if img is None:
        continue
    h, w = img.shape
    ok, corners = cv2.findChessboardCornersSB(img, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ok:
        det.append(dict(name=os.path.relpath(p, a.folder).replace(os.sep, "/"), size=(w, h), corners=corners.astype(np.float32)))
print(f"board detected in {len(det)} / {len(files)}", flush=True)
sizes = {}
for d in det:
    sizes[d["size"]] = sizes.get(d["size"], 0) + 1
(w, h) = max(sizes, key=sizes.get)
dropped_size = [d["name"] for d in det if d["size"] != (w, h)]
det = [d for d in det if d["size"] == (w, h)]
print(f"majority size {w}x{h}: {len(det)} photos kept; dropped for another size: {dropped_size}", flush=True)
if len(det) < 10:
    sys.exit("fewer than 10 usable photos")


def calibrate(items, model="k1k2tang", fix_pp=False):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    if model == "k1":
        flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_ZERO_TANGENT_DIST
    elif model == "k1k2":
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    f0 = 0.69 * max(w, h)
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1.0]])
    rms, K, dist, rv, tv = cv2.calibrateCamera([objp] * len(items), [d["corners"] for d in items], (w, h), K0, None, flags=flags)
    errs = []
    for d, r_, t_ in zip(items, rv, tv):
        pr, _ = cv2.projectPoints(objp, r_, t_, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - d["corners"].reshape(-1, 2)) ** 2, axis=1)))))
    return rms, K, dist, np.array(errs)


# ------------------------------------------------------------------------------------------ outlier removal
items = list(det)
for it in range(3):
    rms, K, dist, errs = calibrate(items)
    bad = errs > 2.5 * np.median(errs)
    if not bad.any():
        break
    print(f"  pass {it}: removing {int(bad.sum())} outlier photo(s) with RMS > {2.5 * np.median(errs):.2f}px: {[items[i]['name'] for i in np.flatnonzero(bad)]}")
    items = [d for d, b in zip(items, bad) if not b]
s = a.max_dim / max(w, h)
fw = lambda K_: float((K_[0, 0] + K_[1, 1]) / 2 * s)

# ------------------------------------------------------------------------------------------ model comparison + stability
models = {}
for m in ("k1", "k1k2", "k1k2tang"):
    r_, K_, d_, e_ = calibrate(items, m)
    models[m] = dict(rms=float(r_), f_working=fw(K_), fx=float(K_[0, 0]), fy=float(K_[1, 1]), cx=float(K_[0, 2]), cy=float(K_[1, 2]), dist=[float(x) for x in d_.ravel()])
rng = np.random.default_rng(0)
fs_half = []
for _ in range(40):
    idx = rng.permutation(len(items))[:len(items) // 2]
    try:
        _, K_, _, _ = calibrate([items[i] for i in idx], "k1k2")
        fs_half.append(fw(K_))
    except cv2.error:
        pass
fs_half = np.array(fs_half)
rms, K, dist, errs = calibrate(items, "k1k2tang")
res = dict(n_candidate=len(files), n_detected=len(det) + len(dropped_size), n_used=len(items), image_size=[w, h], rms_reprojection_px=float(rms),
           fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), dist_k1k2p1p2k3=[float(x) for x in dist.ravel()],
           f_at_working_res=fw(K), aspect_fx_over_fy=float(K[0, 0] / K[1, 1]),
           principal_point_offset_px=[float(K[0, 2] - w / 2), float(K[1, 2] - h / 2)], models=models,
           half_split_f_median=float(np.median(fs_half)), half_split_f_std=float(fs_half.std()), half_split_f_range=[float(fs_half.min()), float(fs_half.max())],
           self_calibrated_f_at_working_res=a.self_calibrated_f_1600,
           per_image_rms={d["name"]: round(float(e), 3) for d, e in zip(items, errs)}, dropped_size=dropped_size)
res["self_calibration_error_pct"] = 100 * (a.self_calibrated_f_1600 / res["f_at_working_res"] - 1)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("\n=== chessboard calibration ===")
print(f"photos used {len(items)} | RMS reprojection {rms:.3f} px | image {w}x{h}")
print(f"f at stored resolution: fx {K[0,0]:.1f}  fy {K[1,1]:.1f} (aspect {K[0,0]/K[1,1]:.4f}) | principal point offset from centre: {K[0,2]-w/2:+.1f}, {K[1,2]-h/2:+.1f} px")
print("distortion k1 k2 p1 p2 k3:", np.round(dist.ravel(), 5))
print("model comparison (f at 1600px / RMS):", {m: (round(v['f_working'], 1), round(v['rms'], 3)) for m, v in models.items()})
print(f"stability over 40 random half-splits: median {np.median(fs_half):.1f}, std {fs_half.std():.1f}, range {fs_half.min():.1f}-{fs_half.max():.1f}")
print(f"\nCHESSBOARD focal length at {a.max_dim}px: {res['f_at_working_res']:.1f}  vs self-calibrated {a.self_calibrated_f_1600:.1f}  ->  self-calibration error {res['self_calibration_error_pct']:+.2f} %")
