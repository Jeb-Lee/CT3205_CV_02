"""YouTube thumbnail (1280x720): one input photo wiping into the dense point cloud rendered from the SAME camera.

    python scripts/make_thumbnail.py --out "C:/Users/jeble/Claude/CT3205_CV_Assignment_2/thumbnail.png"

The cloud is splatted with a z-buffer through the camera's K, R, t from the 16-image dense run, so photo and cloud line
up across the diagonal wipe. Also writes <out>_small.png (320x180) to check legibility at YouTube's list size.
"""
import argparse, os, pickle, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
from export_html import read_ply

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--title", nargs=2, default=["COMPUTER VISION", "ASSIGNMENT 2"])
ap.add_argument("--cam", type=int, default=None, help="camera index to render from (default: middle camera)")
ap.add_argument("--rec", default="results/cache/rec_south-building_40_16_1.pkl")
ap.add_argument("--dense", default="results/dense/dense.ply")
ap.add_argument("--keep", type=float, default=0.6, help="fraction of dense points drawn (lower = more visibly dotty)")
a = ap.parse_args()

W, H = 1280, 720
rec = pickle.load(open(a.rec, "rb"))
ids = sorted(rec.cams)
ci = a.cam if a.cam is not None else ids[len(ids) // 2]
R, t = rec.cams[ci]
K = rec.K
w0, h0 = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))        # working image size (1600x1200)

# ------------------------------------------------------------------ photo from the same camera, at working size
photo = Image.open(os.path.join(rec.ds.image_dir, rec.names[ci])).convert("RGB").resize((w0, h0), Image.LANCZOS)

# ------------------------------------------------------------------ z-buffered splat of the dense cloud
P, C = read_ply(a.dense)
sel = np.random.default_rng(0).random(len(P)) < a.keep
P, C = P[sel], C[sel]
Xc = P @ R.T + t
keep = Xc[:, 2] > 0.05
Xc, C = Xc[keep], C[keep]
uv = Xc[:, :2] / Xc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
z = Xc[:, 2]
rad = np.clip(np.round(1.7 * np.median(z) / z), 1, 3).astype(int)  # nearer points -> bigger splats
pix, zz, cc = [], [], []
for r in (1, 2, 3, 4):
    m = rad >= r
    for dx in range(-r + 1, r):
        for dy in range(-r + 1, r):
            if max(abs(dx), abs(dy)) != r - 1:
                continue
            u = np.round(uv[m, 0]).astype(int) + dx
            v = np.round(uv[m, 1]).astype(int) + dy
            ok = (u >= 0) & (u < w0) & (v >= 0) & (v < h0)
            pix.append(v[ok] * w0 + u[ok]); zz.append(z[m][ok]); cc.append(C[m][ok])
pix, zz, cc = np.concatenate(pix), np.concatenate(zz), np.concatenate(cc)
o = np.lexsort((zz, pix))                                          # per pixel, nearest first
pix, cc = pix[o], cc[o]
first = np.r_[True, pix[1:] != pix[:-1]]
BG = np.array([14, 17, 22], np.uint8)
cloud = np.tile(BG, (h0 * w0, 1))
cloud[pix[first]] = cc[first]
cloud = cloud.reshape(h0, w0, 3).astype(float)
lit = (cloud != BG).any(-1)
cloud[lit] = np.clip(cloud[lit] * 1.3 + 12, 0, 255)
cloud = Image.fromarray(cloud.astype(np.uint8))

# ------------------------------------------------------------------ crop both to 16:9 (same window) and scale
ch = int(w0 * 9 / 16)
y0 = int((h0 - ch) * 0.35)                                         # keep a bit more sky than ground
box = (0, y0, w0, y0 + ch)
photo, cloud = photo.crop(box).resize((W, H), Image.LANCZOS), cloud.crop(box).resize((W, H), Image.LANCZOS)

# ------------------------------------------------------------------ diagonal wipe photo -> cloud with a glowing seam
xx, yy = np.meshgrid(np.arange(W), np.arange(H))
seam = 0.40 * W + (H - yy) * 0.30                                  # leaning seam
alpha = np.clip((xx - seam) / 28 + 0.5, 0, 1)                      # 0 = photo, 1 = cloud
img = np.asarray(photo, float) * (1 - alpha[..., None]) + np.asarray(cloud, float) * alpha[..., None]
glow = np.exp(-((xx - seam) / 5.0) ** 2)[..., None] * np.array([79, 195, 247])
img = Image.fromarray(np.clip(img + glow * 0.9, 0, 255).astype(np.uint8))

# darken the top for the title
grad = np.clip(1 - yy / (H * 0.62), 0, 1) ** 1.4 * 0.78
img = Image.fromarray((np.asarray(img, float) * (1 - grad[..., None]) + np.array(BG) * grad[..., None]).astype(np.uint8))

# ------------------------------------------------------------------ text
d = ImageDraw.Draw(img)
f1 = ImageFont.truetype("C:/Windows/Fonts/ariblk.ttf", 104)
f2 = ImageFont.truetype("C:/Windows/Fonts/ariblk.ttf", 104)
f3 = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 40)


def shadow_text(xy, text, font, fill, blur=10, off=5):
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((xy[0] + off, xy[1] + off), text, font=font, fill=(0, 0, 0, 230))
    img.paste(sh.filter(ImageFilter.GaussianBlur(blur)), (0, 0), sh.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).text(xy, text, font=font, fill=fill, stroke_width=3, stroke_fill=(10, 12, 16))


shadow_text((48, 26), a.title[0], f1, (245, 245, 245))
shadow_text((48, 146), a.title[1], f2, (79, 195, 247))
# labels on each side of the seam
for (x, y, s) in [(40, H - 78, "2D PHOTOS"), (None, H - 78, "3D POINT CLOUD")]:
    tw = d.textlength(s, font=f3)
    x = W - 40 - tw if x is None else x
    ImageDraw.Draw(img).rounded_rectangle((x - 14, y - 8, x + tw + 14, y + 52), 12, fill=(14, 17, 22))
    ImageDraw.Draw(img).text((x, y), s, font=f3, fill=(245, 245, 245))

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
img.save(a.out)
img.resize((320, 180), Image.LANCZOS).save(os.path.splitext(a.out)[0] + "_small.png")
print("camera", ci, rec.names[ci], "->", a.out)
