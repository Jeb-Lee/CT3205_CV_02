"""Review a folder of chessboard calibration photos (HEIC/JPEG, sub-folders allowed) before calibrating.

    python scripts/check_calibration_photos.py --folder Datasets/Calibration --cols 9 --rows 6
Per photo: stored size, EXIF (lens, focal length, focus distance, orientation), 'edited' flag (iPhone IMG_Exxxx = edited copy),
board found?, board width as % of the frame, tilt of the board plane, position in the frame. Summary per sub-folder.
Writes results/own/calib_photo_review.csv and a contact sheet of the detections.
"""
import argparse, os, sys
import cv2
import numpy as np
import pandas as pd
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()
ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
a = ap.parse_args()
pattern = (a.cols, a.rows)

rows, sheet = [], []
for dp, _, fs in sorted(os.walk(a.folder)):
    for f in sorted(fs):
        if not f.lower().endswith((".heic", ".heif", ".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(dp, f)
        im = Image.open(path)
        ex = im.getexif()
        ifd = ex.get_ifd(0x8769)
        gray = np.array(im.convert("L"))
        h, w = gray.shape
        r = dict(folder=os.path.relpath(dp, a.folder), file=f, edited=f.upper().startswith("IMG_E"), width=w, height=h,
                 orientation=ex.get(0x0112), model=str(ex.get(0x0110)), focal_mm=float(ifd.get(0x920A, 0) or 0),
                 focal35=ifd.get(0xA405), lens=str(ifd.get(0xA434)), subject_dist_m=float(ifd.get(0x9206, 0) or 0) if ifd.get(0x9206) is not None else None)
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        r["board_found"] = bool(ok)
        if ok:
            c = corners.reshape(-1, 2)
            grid = c.reshape(a.rows, a.cols, 2)
            top = np.linalg.norm(grid[0, -1] - grid[0, 0]); bot = np.linalg.norm(grid[-1, -1] - grid[-1, 0])
            left = np.linalg.norm(grid[-1, 0] - grid[0, 0]); right = np.linalg.norm(grid[-1, -1] - grid[0, -1])
            r["board_width_pct"] = round(100 * max(top, bot) / w, 1)
            r["perspective_lr_ratio"] = round(float(left / right), 3)          # != 1 -> board turned about the vertical axis
            r["perspective_tb_ratio"] = round(float(top / bot), 3)             # != 1 -> board tilted about the horizontal axis
            r["center_x_pct"] = round(100 * c[:, 0].mean() / w, 1); r["center_y_pct"] = round(100 * c[:, 1].mean() / h, 1)
            vis = cv2.cvtColor(cv2.resize(gray, (w // 6, h // 6)), cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern, (corners / 6).astype(np.float32), True)
        else:
            vis = cv2.cvtColor(cv2.resize(gray, (w // 6, h // 6)), cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, "NOT FOUND", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        cv2.putText(vis, f"{r['folder']}/{f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        sheet.append(cv2.resize(vis, (300, 225)))
        rows.append(r)

df = pd.DataFrame(rows)
os.makedirs("results/own", exist_ok=True)
df.to_csv("results/own/calib_photo_review.csv", index=False)
while len(sheet) % 8:
    sheet.append(np.zeros((225, 300, 3), np.uint8))
cv2.imwrite("results/own/calib_photo_review.jpg", np.vstack([np.hstack(sheet[i:i + 8]) for i in range(0, len(sheet), 8)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(f"{len(df)} photos; stored sizes: {df.groupby(['width', 'height']).size().to_dict()}")
print("orientation tags:", df.orientation.value_counts().to_dict(), "| models:", df.model.value_counts().to_dict(), "| focal35:", df.focal35.value_counts().to_dict())
print("\nper folder (found / total, edited copies, median board width %):")
g = df.groupby("folder").agg(total=("file", "size"), found=("board_found", "sum"), edited=("edited", "sum"), board_w_pct=("board_width_pct", "median"))
print(g.to_string())
print("\nedited (IMG_E) vs original pairs with the same number:")
nums = df.assign(num=df.file.str.extract(r"(\d+)")[0]).groupby("num").file.apply(list)
pairs = {k: v for k, v in nums.items() if len(v) > 1}
print(pairs if pairs else "none")
