"""Create a printable chessboard for calibration: 10 x 7 squares (= 9 x 6 inner corners), 25 mm squares, A4 landscape.

    python scripts/make_chessboard.py
Writes docs/calibration/chessboard_10x7_A4.pdf (print at 100 % / 'actual size', NOT 'fit to page') and a PNG preview.
After printing, MEASURE the real square size with a ruler across several squares (printers scale slightly) and pass it as
--square_mm to calibrate_chessboard.py.
"""
import os
import numpy as np
from PIL import Image

DPI = 300
mm = lambda x: x / 25.4 * DPI
W, H = int(round(mm(297))), int(round(mm(210)))                 # A4 landscape
sq = int(round(mm(25)))                                          # 25 mm squares (about 295 px at 300 dpi)
cols, rows = 10, 7
img = np.full((H, W), 255, np.uint8)
x0, y0 = (W - cols * sq) // 2, (H - rows * sq) // 2
for r in range(rows):
    for c in range(cols):
        if (r + c) % 2 == 0:
            img[y0 + r * sq:y0 + (r + 1) * sq, x0 + c * sq:x0 + (c + 1) * sq] = 0
os.makedirs("docs/calibration", exist_ok=True)
im = Image.fromarray(img)
im.save("docs/calibration/chessboard_10x7_A4.pdf", "PDF", resolution=DPI)
im.resize((W // 3, H // 3)).save("docs/calibration/chessboard_10x7_A4_preview.png")
print(f"board {cols * sq / DPI * 25.4:.1f} x {rows * sq / DPI * 25.4:.1f} mm, square {sq / DPI * 25.4:.2f} mm nominal; wrote docs/calibration/chessboard_10x7_A4.pdf")
