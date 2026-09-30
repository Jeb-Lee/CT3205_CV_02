"""Dataset abstraction + image loading / synthetic degradation.

Two kinds of dataset are supported:
  * COLMAP datasets  (Datasets/<name>/{images,sparse}) - intrinsics + ground-truth poses known
  * custom datasets  (Datasets/<name>/images + optional intrinsics.json) - your own photos.
    Intrinsics come from intrinsics.json ({"fx":..,"fy":..,"cx":..,"cy":..,"dist":[k1,k2,p1,p2,k3]})
    or, failing that, are estimated from EXIF focal length / a 1.2*max(w,h) guess.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import io_colmap as ioc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "Datasets")


@dataclass
class Dataset:
    name: str
    image_dir: str
    names: list
    K: np.ndarray            # intrinsics at ORIGINAL resolution
    dist: np.ndarray         # OpenCV distortion (k1,k2,p1,p2,k3)
    size: tuple              # (width, height) original
    gt_poses: dict = field(default_factory=dict)   # name -> (R, t) world->cam
    gt_points: np.ndarray | None = None
    gt_colors: np.ndarray | None = None

    def path(self, name):
        return os.path.join(self.image_dir, name)


def load_dataset(name: str) -> Dataset:
    root = os.path.join(DATA_DIR, name)
    image_dir = os.path.join(root, "images")
    names = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    sparse = os.path.join(root, "sparse")
    if os.path.isfile(os.path.join(sparse, "cameras.txt")):
        cams = ioc.read_cameras(os.path.join(sparse, "cameras.txt"))
        imgs = ioc.read_images(os.path.join(sparse, "images.txt"))
        cam = cams[next(iter(imgs.values()))["cam_id"]]
        K, dist = ioc.camera_to_K_dist(cam)
        xyz, rgb, _ = ioc.read_points3D(os.path.join(sparse, "points3D.txt"))
        gt = {n: (v["R"], v["t"]) for n, v in imgs.items()}
        return Dataset(name, image_dir, names, K, dist, (cam["width"], cam["height"]), gt, xyz, rgb)
    return _load_custom(name, image_dir, names, root)


def _load_custom(name, image_dir, names, root):
    im = cv2.imread(os.path.join(image_dir, names[0]))
    h, w = im.shape[:2]
    jpath = os.path.join(root, "intrinsics.json")
    if os.path.isfile(jpath):
        j = json.load(open(jpath))
        K = np.array([[j["fx"], 0, j["cx"]], [0, j["fy"], j["cy"]], [0, 0, 1.0]])
        dist = np.array(j.get("dist", [0, 0, 0, 0, 0]), dtype=float)
    else:
        f = _exif_focal_px(os.path.join(image_dir, names[0]), w, h) or 1.2 * max(w, h)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        dist = np.zeros(5)
        print(f"[datasets] no intrinsics.json for {name}; using f={f:.1f}px (EXIF/guess). "
              f"Calibrate with a checkerboard for best results.")
    return Dataset(name, image_dir, names, K, dist, (w, h))


def _exif_focal_px(path, w, h):
    """Focal length in pixels from 35mm-equivalent EXIF tag (works for most phones)."""
    try:
        from PIL import Image
        ex = Image.open(path).getexif().get_ifd(0x8769)
        f35 = ex.get(0xA405)  # FocalLengthIn35mmFilm
        if f35:
            return float(f35) / 36.0 * max(w, h)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------------------
# Image loading and controlled degradation (Experiment D)
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Degrade:
    blur_sigma: float = 0.0     # Gaussian blur sigma (px)
    noise_sigma: float = 0.0    # additive Gaussian noise (grey levels)
    contrast: float = 1.0       # multiply around mid-grey
    brightness: float = 0.0     # additive offset (grey levels)
    gamma: float = 1.0          # >1 darker, <1 brighter

    def key(self):
        return f"b{self.blur_sigma}_n{self.noise_sigma}_c{self.contrast}_l{self.brightness}_g{self.gamma}"

    def is_identity(self):
        return self == Degrade()


def load_image(ds: Dataset, name: str, max_dim: int, degrade: Degrade | None = None, seed=0):
    """Load, resize (longest side = max_dim), undistort, optionally degrade.

    Returns (bgr uint8, K_scaled). After undistortion the image follows an ideal pinhole model
    with intrinsics K_scaled, so all downstream geometry can ignore lens distortion.
    """
    img = cv2.imread(ds.path(name), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    K = ds.K.copy()
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        K[:2] *= s
    if np.any(ds.dist != 0):
        img = cv2.undistort(img, K, ds.dist, None, K)
    if degrade is not None and not degrade.is_identity():
        img = apply_degradation(img, degrade, seed)
    return img, K


def apply_degradation(img, d: Degrade, seed=0):
    x = img.astype(np.float32)
    if d.gamma != 1.0:
        x = 255.0 * (x / 255.0) ** d.gamma
    if d.contrast != 1.0 or d.brightness != 0.0:
        x = (x - 127.5) * d.contrast + 127.5 + d.brightness
    if d.blur_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), d.blur_sigma)
    if d.noise_sigma > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0, d.noise_sigma, x.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def select_images(ds: Dataset, start=0, count=12, stride=1):
    """Pick `count` images from the name-sorted sequence starting at `start` with `stride`."""
    return ds.names[start:start + count * stride:stride][:count]
