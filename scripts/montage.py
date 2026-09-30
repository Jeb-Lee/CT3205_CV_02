"""Contact sheet of every k-th image of a dataset (to understand the capture pattern before choosing subsets)."""
import argparse, os, sys, cv2, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="person-hall"); ap.add_argument("--step", type=int, default=15)
ap.add_argument("--cols", type=int, default=6); ap.add_argument("--out", default="results/person_hall/contact_sheet.jpg")
a = ap.parse_args()
ds = D.load_dataset(a.dataset)
tiles = []
for k in range(0, len(ds.names), a.step):
    im, _ = D.load_image(ds, ds.names[k], 420)
    im = cv2.resize(im, (420, 280)); cv2.putText(im, f"{k}: {ds.names[k]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    tiles.append(im)
while len(tiles) % a.cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(len(ds.names), "images;", a.out)
