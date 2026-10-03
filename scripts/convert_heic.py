"""Convert iPhone HEIC/HEIF photos to JPEG (geometry unchanged: same pixels, same size; quality 95).

    python scripts/convert_heic.py Datasets/Calibration            # writes IMG_xxxx.jpg next to each .heic (originals kept)
Needed only if you want JPEGs; scripts/calibrate_chessboard.py reads HEIC directly. EXIF (focal length, exposure ...) is copied.
"""
import os
import sys

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()
folder = sys.argv[1]
n = 0
for f in sorted(os.listdir(folder)):
    if f.lower().endswith((".heic", ".heif")):
        im = Image.open(os.path.join(folder, f))
        exif = im.info.get("exif")
        out = os.path.join(folder, os.path.splitext(f)[0] + ".jpg")
        im.convert("RGB").save(out, "JPEG", quality=95, exif=exif if exif else b"")
        n += 1
print(f"converted {n} HEIC file(s) in {folder}; sizes e.g. {Image.open(out).size if n else '-'}")
