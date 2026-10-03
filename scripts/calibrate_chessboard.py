"""Chessboard calibration of the phone camera (ground truth for our self-calibrated focal length).

    python scripts/make_chessboard.py                                   # creates the printable board (once)
    python scripts/calibrate_chessboard.py --folder Datasets/Calibration --cols 9 --rows 6 --square_mm 25
(cols x rows = number of INNER corners; the printable board has 10 x 7 squares = 9 x 6 inner corners)

Reports focal length and principal point at the stored image resolution AND at the 1600 px working resolution we use in the
pipeline, radial/tangential distortion, the RMS reprojection error, per-image errors (outliers flagged) and the comparison
with the self-calibrated value (1154 px at 1600 px). Writes results/own/chessboard_calibration.json and an image of the detected
corners for the report.
"""
import argparse, json, os, sys
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square_mm", type=float, default=25.0)
ap.add_argument("--self_calibrated_f_1600", type=float, default=1154.0)
ap.add_argument("--max_dim", type=int, default=1600)
ap.add_argument("--out", default="results/own/chessboard_calibration.json")
a = ap.parse_args()

pattern = (a.cols, a.rows)
objp = np.zeros((a.cols * a.rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_mm
files = sorted(f for f in os.listdir(a.folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif")))
obj_pts, img_pts, used, sizes = [], [], [], set()


def load_gray(path):
    """JPEG/PNG through OpenCV (stored pixels, orientation tag ignored, like our pipeline); iPhone HEIC/HEIF through pillow-heif."""
    if path.lower().endswith((".heic", ".heif")):
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        return np.array(Image.open(path).convert("L"))          # libheif applies the stored rotation: landscape shots stay landscape
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)


for f in files:
    img = load_gray(os.path.join(a.folder, f))
    if img is None:
        continue
    h, w = img.shape
    ok, corners = cv2.findChessboardCornersSB(img, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        print(f"  {f}: board NOT found"); continue
    sizes.add((w, h)); obj_pts.append(objp); img_pts.append(corners.astype(np.float32)); used.append(f)
if len(used) < 8:
    sys.exit(f"only {len(used)} usable images (need >= 8, ideally 20+)")
if len(sizes) > 1:
    sys.exit(f"images have different sizes {sizes}: shoot all calibration photos at the same resolution as the object photos")
(w, h), = sizes
flags = cv2.CALIB_FIX_K3                                   # k1, k2 + tangential are enough for a phone main camera
f0 = 0.69 * max(w, h) * 1.0
K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1.0]])
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), K0, None, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)
errs = []
for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
    proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
    errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))))
errs = np.array(errs)
s = a.max_dim / max(w, h)
res = dict(images_found=len(used), images_total=len(files), image_size=[w, h], rms_reprojection_px=float(rms),
           fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), dist_k1k2p1p2k3=[float(x) for x in dist.ravel()],
           f_at_working_res=float((K[0, 0] + K[1, 1]) / 2 * s), self_calibrated_f_at_working_res=a.self_calibrated_f_1600,
           aspect_fx_over_fy=float(K[0, 0] / K[1, 1]), principal_point_offset_px=[float(K[0, 2] - w / 2), float(K[1, 2] - h / 2)],
           per_image_rms=dict(zip(used, [round(float(e), 3) for e in errs])), worst_images=[used[i] for i in np.argsort(-errs)[:3]])
res["self_calibration_error_pct"] = 100 * (a.self_calibrated_f_1600 / res["f_at_working_res"] - 1)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_image_rms"}, indent=1))
print(f"\nchessboard focal length at {a.max_dim}px: {res['f_at_working_res']:.1f}  vs self-calibrated {a.self_calibrated_f_1600:.1f}  ->  self-calibration error {res['self_calibration_error_pct']:+.2f} %")
