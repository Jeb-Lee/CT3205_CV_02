"""Why is the chessboard RMS high? Re-calibrate with subsets (oblique views removed, no distortion model, frontal only)."""
import os, sys, pickle
import cv2, numpy as np
import pillow_heif
from PIL import Image
pillow_heif.register_heif_opener()
folder = "Datasets/Calibration"; pattern = (9, 6); sq = 25.0
objp = np.zeros((54, 3), np.float32); objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * sq
cache = "results/own/_calib_corners.pkl"
if os.path.exists(cache):
    det = pickle.load(open(cache, "rb"))
else:
    det = []
    for dp, dn, fs in os.walk(folder):
        dn[:] = [d for d in dn if d not in ("measurement", "misc")]
        for f in sorted(fs):
            if not f.lower().endswith(".heic") or f.upper().startswith("IMG_E"):
                continue
            g = np.array(Image.open(os.path.join(dp, f)).convert("L"))
            ok, c = cv2.findChessboardCornersSB(g, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ok and g.shape == (4284, 5712):
                det.append(dict(name=f"{os.path.basename(dp)}/{f}", corners=c.reshape(-1, 2).astype(np.float32)))
    pickle.dump(det, open(cache, "wb"))
w, h = 5712, 4284
for d in det:                                      # tilt measures from the detected grid
    g = d["corners"].reshape(6, 9, 2)
    top, bot = np.linalg.norm(g[0, -1] - g[0, 0]), np.linalg.norm(g[-1, -1] - g[-1, 0])
    left, right = np.linalg.norm(g[-1, 0] - g[0, 0]), np.linalg.norm(g[-1, -1] - g[0, -1])
    d["tilt"] = max(abs(top / bot - 1), abs(left / right - 1)); d["width_pct"] = 100 * max(top, bot) / w


def calib(items, flags_extra=0, label=""):
    f0 = 0.69 * w
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1.0]])
    rms, K, dist, rv, tv = cv2.calibrateCamera([objp] * len(items), [d["corners"].reshape(-1, 1, 2) for d in items], (w, h), K0, None,
                                               flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | flags_extra)
    errs = []
    for d, r_, t_ in zip(items, rv, tv):
        pr, _ = cv2.projectPoints(objp, r_, t_, K, dist)
        errs.append(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - d["corners"]) ** 2, 1))))
    print(f"{label:55s} n={len(items):2d}  RMS {rms:5.2f}px  median-per-image {np.median(errs):5.2f}  f@1600 {(K[0,0]+K[1,1])/2*1600/w:7.1f}  "
          f"k1 {dist[0,0]:+.3f} k2 {dist[0,1]:+.3f}  pp-offset ({K[0,2]-w/2:+.0f},{K[1,2]-h/2:+.0f})")
    return errs, K, dist


NOD = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_ZERO_TANGENT_DIST
errs, K, dist = calib(det, 0, "all 40, k1 k2 p1 p2")
calib(det, NOD, "all 40, NO distortion model")
t = np.array([d["tilt"] for d in det])
print(f"\nper-image RMS vs obliqueness: correlation {np.corrcoef(errs, t)[0,1]:+.2f}; tilt quartiles {np.percentile(t,[25,50,75]).round(2)}")
order = np.argsort(-np.array(errs))
print("worst 6 photos:", [(det[i]["name"], round(float(errs[i]), 1), round(float(t[i]), 2)) for i in order[:6]])
print("best 6 photos :", [(det[i]["name"], round(float(errs[i]), 1), round(float(t[i]), 2)) for i in order[-6:]])
print()
for thr in (0.5, 0.3, 0.15):
    sub = [d for d in det if d["tilt"] < thr]
    if len(sub) >= 8:
        calib(sub, 0, f"tilt < {thr} (less oblique)")
        calib(sub, NOD, f"tilt < {thr}, no distortion model")
sub = [d for d in det if d["name"].startswith("straight")]
calib(sub, NOD, "folder 'straight' only, no distortion model")
sub = [d for d in det if d["name"].startswith("off_angle") or d["name"].startswith("far")]
print("(off_angle + far only:", len(sub), "photos, too few to calibrate alone)")
