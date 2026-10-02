"""Incremental multi-view reconstruction (Structure-from-Motion) - Assignment 1.4 / 1.6.

Pipeline
--------
1. extract features for every image                       (features.py)
2. match all pairs inside a window, verify with RANSAC-E   (matching.py)
3. initialise from the best two-view pair: E -> (R,t) by cheirality, triangulate  (geometry.py)
4. repeatedly register the image sharing most 2D-3D correspondences with the model via PnP+RANSAC,
   then triangulate new points with every already-registered image
5. bundle adjustment after each registration (optional) and a final outlier filtering step

World frame = camera of the initial pair's first image (P1 = K[I|0]); scale = |baseline| of that pair = 1.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

from . import features as F
from . import matching as M
from . import geometry as G
from . import datasets as D
from .bundle import bundle_adjust


@dataclass
class SfMConfig:
    detector: str = "rootsift"        # sift|rootsift|orb|akaze|brisk | disk (learned, GPU)
    matcher: str = "bf"               # bf (CPU brute force) | gpu (same algorithm on GPU) | lightglue (learned, needs detector=disk)
    nfeatures: int = 8000
    max_dim: int = 1600
    ratio: float = 0.8
    cross_check: bool = True
    ransac_thr: float = 1.0        # px, essential-matrix RANSAC
    window: int | None = None      # match image i with i+1..i+window (None = all pairs)
    min_pair_inliers: int = 40     # a pair counts as "verified" / may seed the reconstruction
    min_pair_use: int = 25         # pairs below this are treated as unrelated images (RANSAC noise)
    init_min_angle: float = 3.0    # deg, median parallax needed for the initial pair
    init_candidates: int = 12
    tri_min_angle: float = 1.5     # deg, min parallax for a new 3D point
    max_reproj: float = 2.0        # px, max reprojection error of accepted points
    pnp_thr: float = 3.0           # px, RANSAC threshold for PnP
    min_pnp: int = 20
    use_ba: bool = True
    refine_focal: bool = False     # self-calibration: refine ONE shared focal length in bundle adjustment (needs >= 8 cameras)
    ba_every: int = 1              # run BA after every k-th registration
    seed: int = 0


class Reconstruction:
    """Holds the state of one SfM run; call run() then read the public result arrays."""

    def __init__(self, ds: D.Dataset, names: list, cfg: SfMConfig, degrade: D.Degrade | None = None,
                 cache_dir: str | None = None, verbose=True):
        self.ds, self.names, self.cfg = ds, list(names), cfg
        self.degrade = degrade or D.Degrade()
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.n = len(names)
        self.K = None
        self.kps, self.descs = [], []
        self.matches = {}           # (i,j) i<j -> dict(i1,i2 verified idx, E, n_raw, n_ratio, n_mutual, n_inl)
        self.cams = {}              # img -> (R, t)
        self.points = {}            # pid -> xyz
        self.tracks = {}            # pid -> {img: kp_idx}
        self.kp2pt = [dict() for _ in range(self.n)]
        self._next_pid = 0
        self.log = dict(events=[], timing={})
        self.init_pair = None

    def say(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------ features
    def compute_features(self):
        t0 = time.time()
        for i, name in enumerate(self.names):
            cache = self._cache_path(name)
            if cache and os.path.isfile(cache):
                z = np.load(cache, allow_pickle=True)
                kp, desc = z["kp"], z["desc"]
                K = self._K_from_size(name)       # NOT the cached K: it may belong to another focal length (focal sweep)
                desc = None if desc.shape == () else desc
            else:
                img, K = D.load_image(self.ds, name, self.cfg.max_dim, self.degrade, seed=i)
                if self.cfg.detector == "disk":
                    kp, desc = self._learned().extract(img)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    kp, desc = F.extract(gray, self.cfg.detector, self.cfg.nfeatures)
                if cache:
                    os.makedirs(os.path.dirname(cache), exist_ok=True)
                    np.savez_compressed(cache, kp=kp, desc=desc if desc is not None else np.array(0), K=K)
            self.kps.append(kp)
            self.descs.append(desc)
            self.K = K
        self.log["timing"]["features"] = time.time() - t0
        self.log["n_features"] = [len(k) for k in self.kps]
        self.say(f"[features] {self.cfg.detector}: mean {np.mean(self.log['n_features']):.0f} kp/img "
                 f"({self.log['timing']['features']:.1f}s)")

    def _K_from_size(self, name):
        """Camera matrix for the resized image, from the stored size only (no pixel decoding)."""
        from PIL import Image
        w, h = Image.open(self.ds.path(name)).size
        s = min(1.0, self.cfg.max_dim / max(h, w))
        K = self.ds.K.copy()
        K[:2] *= max(h, w) / max(self.ds.size)
        K[:2] *= s
        return K

    _LEARNED = None

    def _learned(self):
        if Reconstruction._LEARNED is None:           # models are loaded once per process
            from .gpu_matching import DiskLightGlue
            Reconstruction._LEARNED = DiskLightGlue(self.cfg.nfeatures)
        return Reconstruction._LEARNED

    def _cache_path(self, name):
        if not self.cache_dir:
            return None
        c = self.cfg
        key = f"v2io_{self.ds.name}_{name}_{c.max_dim}_{c.detector}_{c.nfeatures}_{self.degrade.key()}"
        return os.path.join(self.cache_dir, re.sub(r"[^A-Za-z0-9_\-]", "_", key) + ".npz")

    # ------------------------------------------------------------------ matching
    def match_all(self):
        t0 = time.time()
        norm = F.norm_for(self.cfg.detector) if self.cfg.detector != "disk" else cv2.NORM_L2
        w = self.cfg.window or self.n
        for i in range(self.n):
            for j in range(i + 1, min(self.n, i + w + 1)):
                self._match(i, j, norm)
        self.log["timing"]["matching"] = time.time() - t0
        n_ok = sum(1 for m in self.matches.values() if m["n_inl"] >= self.cfg.min_pair_inliers)
        self.say(f"[matching] {len(self.matches)} pairs, {n_ok} verified (>= {self.cfg.min_pair_inliers} inliers) "
                 f"({self.log['timing']['matching']:.1f}s)")

    def _match(self, i, j, norm):
        rec = dict(n_raw=0, n_ratio=0, n_mutual=0, n_inl=0, E=None,
                   i1=np.zeros(0, int), i2=np.zeros(0, int))
        if self.descs[i] is not None and self.descs[j] is not None and len(self.kps[i]) > 8 and len(self.kps[j]) > 8:
            if self.cfg.matcher == "lightglue":
                size = lambda k: (2 * self.K[0, 2], 2 * self.K[1, 2])
                i1, i2, st = self._learned().match(self.kps[i], self.descs[i], self.kps[j], self.descs[j], size(i), size(j))
            elif self.cfg.matcher == "gpu":
                from .gpu_matching import match_pair_gpu
                i1, i2, st = match_pair_gpu(self.descs[i], self.descs[j], self.cfg.ratio, self.cfg.cross_check)
            else:
                i1, i2, st = M.match_pair(self.descs[i], self.descs[j], norm, self.cfg.ratio, self.cfg.cross_check)
            rec.update(st)
            E, mask = M.verify_essential(self.kps[i][i1], self.kps[j][i2], self.K, self.cfg.ransac_thr)
            rec.update(E=E, n_inl=int(mask.sum()), i1=i1[mask], i2=i2[mask])
        self.matches[(i, j)] = rec

    def pair_matches(self, i, j):
        """Verified matches oriented as (kp indices in i, kp indices in j)."""
        empty = (np.zeros(0, int), np.zeros(0, int))
        m = self.matches.get((i, j) if i < j else (j, i))
        # pairs with only a handful of "inliers" are RANSAC noise (8-point fits always find ~8): ignore them
        if not m or m["n_inl"] < self.cfg.min_pair_use:
            return empty
        return (m["i1"], m["i2"]) if i < j else (m["i2"], m["i1"])

    # ------------------------------------------------------------------ initialisation
    def initialize(self):
        cands = sorted(((m["n_inl"], k) for k, m in self.matches.items() if m["n_inl"] >= self.cfg.min_pair_inliers),
                       reverse=True)[:self.cfg.init_candidates]
        best = None
        for n_inl, (i, j) in cands:
            m = self.matches[(i, j)]
            p1, p2 = self.kps[i][m["i1"]], self.kps[j][m["i2"]]
            R, t, counts, X = G.recover_pose_cheirality(m["E"], self.K, p1, p2)
            C2 = G.camera_center(R, t)
            ang = G.triangulation_angle_deg(np.zeros(3), C2, X)
            e1, z1 = G.reproj_error(self.K, np.eye(3), np.zeros(3), X, p1)
            e2, z2 = G.reproj_error(self.K, R, t, X, p2)
            ok = (z1 > 0) & (z2 > 0) & (e1 < self.cfg.max_reproj) & (e2 < self.cfg.max_reproj)
            if ok.sum() < 20:
                continue
            med = float(np.median(ang[ok]))
            score = ok.sum() * min(1.0, med / max(self.cfg.init_min_angle, 1e-6))
            self.say(f"  init candidate ({i},{j}): inl={n_inl} good={ok.sum()} median parallax={med:.2f} deg "
                     f"cheirality counts={counts}")
            if best is None or score > best[0]:
                best = (score, i, j, R, t, med, counts)
        if best is None:
            raise RuntimeError("no valid initial pair (too few matches / too little parallax)")
        _, i, j, R, t, med, counts = best
        self.cams[i] = (np.eye(3), np.zeros(3))
        self.cams[j] = (R, t)
        self.init_pair = (i, j)
        n0 = self._triangulate_pair(i, j)
        self.log["events"].append(dict(type="init", pair=[i, j], median_parallax=med, cheirality=counts, points=n0))
        self.say(f"[init] pair ({self.names[i]}, {self.names[j]}): {n0} points, median parallax {med:.2f} deg, "
                 f"cheirality counts (per candidate) {counts}")

    # ------------------------------------------------------------------ triangulation
    def _add_point(self, xyz, obs: dict):
        pid = self._next_pid
        self._next_pid += 1
        self.points[pid] = xyz
        self.tracks[pid] = dict(obs)
        for im, kp in obs.items():
            self.kp2pt[im][kp] = pid
        return pid

    def _triangulate_pair(self, i, j):
        """Create new points from matches of (i, j) that are not yet in the model, and extend tracks."""
        a, b = self.pair_matches(i, j)
        if len(a) == 0:
            return 0
        Ri, ti = self.cams[i]
        Rj, tj = self.cams[j]
        pa, pb = self.kps[i][a], self.kps[j][b]
        new_a, new_b = [], []
        added = 0
        for ka, kb in zip(a, b):
            pida, pidb = self.kp2pt[i].get(ka), self.kp2pt[j].get(kb)
            if pida is None and pidb is None:
                new_a.append(ka); new_b.append(kb)
            elif pida is not None and pidb is None:
                added += self._try_extend(pida, j, kb)
            elif pida is None and pidb is not None:
                added += self._try_extend(pidb, i, ka)
        if new_a:
            new_a, new_b = np.array(new_a), np.array(new_b)
            xa, xb = self.kps[i][new_a], self.kps[j][new_b]
            X = G.triangulate_dlt(Ri, ti, Rj, tj, G.to_normalized(xa, self.K), G.to_normalized(xb, self.K))
            ea, za = G.reproj_error(self.K, Ri, ti, X, xa)
            eb, zb = G.reproj_error(self.K, Rj, tj, X, xb)
            ang = G.triangulation_angle_deg(G.camera_center(Ri, ti), G.camera_center(Rj, tj), X)
            ok = (za > 0) & (zb > 0) & (ea < self.cfg.max_reproj) & (eb < self.cfg.max_reproj) \
                 & (ang > self.cfg.tri_min_angle) & np.isfinite(X).all(axis=1)
            for k in np.flatnonzero(ok):
                self._add_point(X[k], {i: int(new_a[k]), j: int(new_b[k])})
                added += 1
        return added

    def _try_extend(self, pid, img, kp):
        if img in self.tracks[pid]:
            return 0
        R, t = self.cams[img]
        e, z = G.reproj_error(self.K, R, t, self.points[pid][None], self.kps[img][kp][None])
        if z[0] > 0 and e[0] < self.cfg.max_reproj:
            self.tracks[pid][img] = int(kp)
            self.kp2pt[img][int(kp)] = pid
            return 0  # extends a track, no new point
        return 0

    # ------------------------------------------------------------------ registration
    def _correspondences(self, u):
        """2D-3D correspondences of unregistered image u via its matches to registered images."""
        d = {}
        for r in self.cams:
            a, b = self.pair_matches(u, r)
            for ku, kr in zip(a, b):
                pid = self.kp2pt[r].get(kr)
                if pid is not None:
                    d.setdefault(int(ku), {}).setdefault(pid, 0)
                    d[int(ku)][pid] += 1
        return {ku: max(v.items(), key=lambda kv: kv[1])[0] for ku, v in d.items()}

    def register_next(self):
        remaining = [u for u in range(self.n) if u not in self.cams]
        scored = [(len(c := self._correspondences(u)), u, c) for u in remaining]
        scored.sort(key=lambda s: -s[0])
        for n_corr, u, corr in scored:
            if n_corr < self.cfg.min_pnp:
                break
            kus = np.array(list(corr.keys()))
            pids = [corr[k] for k in kus]
            obj = np.array([self.points[p] for p in pids], dtype=np.float64)
            img = self.kps[u][kus].astype(np.float64)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, iterationsCount=2000,
                                                     reprojectionError=self.cfg.pnp_thr, confidence=0.9999,
                                                     flags=cv2.SOLVEPNP_EPNP)
            if not ok or inl is None or len(inl) < self.cfg.min_pnp:
                continue
            inl = inl.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
            R = cv2.Rodrigues(rvec)[0]
            t = tvec.ravel()
            self.cams[u] = (R, t)
            for k in inl:
                pid = pids[k]
                if u not in self.tracks[pid] and int(kus[k]) not in self.kp2pt[u]:
                    self.tracks[pid][u] = int(kus[k])
                    self.kp2pt[u][int(kus[k])] = pid
            n_new = sum(self._triangulate_pair(u, r) for r in list(self.cams) if r != u)
            self.log["events"].append(dict(type="register", image=self.names[u], correspondences=n_corr,
                                           pnp_inliers=int(len(inl)), new_points=int(n_new)))
            self.say(f"[register] {self.names[u]}: {n_corr} 2D-3D corr, {len(inl)} PnP inliers, +{n_new} points "
                     f"(total {len(self.points)}, cams {len(self.cams)})")
            return True
        return False

    # ------------------------------------------------------------------ refinement
    def _observations(self):
        return [(im, pid, tuple(self.kps[im][kp])) for pid, tr in self.tracks.items()
                for im, kp in tr.items() if im in self.cams]

    def bundle_adjust(self):
        if len(self.cams) < 2 or not self.points:
            return
        t0 = time.time()
        fixed = self.init_pair[0]
        refine = self.cfg.refine_focal and len(self.cams) >= 8
        res = bundle_adjust(self.K, self.cams, self.points, self._observations(), fixed=fixed, refine_f=refine)
        if refine:
            cams, pts, (r0, r1), fs = res
            self.K = self.K.copy(); self.K[0, 0] *= fs; self.K[1, 1] *= fs
            self.log.setdefault("focal_history", []).append(float(self.K[0, 0]))
        else:
            cams, pts, (r0, r1) = res
        self.cams, self.points = cams, pts
        self.log["events"].append(dict(type="ba", cams=len(cams), points=len(pts), rmse_before=r0, rmse_after=r1,
                                       seconds=time.time() - t0))
        self.say(f"[BA] {len(cams)} cams, {len(pts)} pts: RMSE {r0:.3f} -> {r1:.3f} px ({time.time() - t0:.1f}s)")

    def filter_points(self, max_err=None):
        """Drop observations with large error / negative depth, then points with < 2 views."""
        max_err = max_err or self.cfg.max_reproj
        removed = 0
        for pid in list(self.tracks):
            X = self.points[pid]
            for im, kp in list(self.tracks[pid].items()):
                if im not in self.cams:
                    continue
                R, t = self.cams[im]
                e, z = G.reproj_error(self.K, R, t, X[None], self.kps[im][kp][None])
                if z[0] <= 0 or e[0] > max_err:
                    del self.tracks[pid][im]
                    self.kp2pt[im].pop(kp, None)
            if len(self.tracks[pid]) < 2:
                for im, kp in self.tracks[pid].items():
                    self.kp2pt[im].pop(kp, None)
                del self.tracks[pid]
                del self.points[pid]
                removed += 1
        return removed

    # ------------------------------------------------------------------ driver
    def run(self):
        t0 = time.time()
        self.compute_features()
        self.match_all()
        self.initialize()
        if self.cfg.use_ba:
            self.bundle_adjust()
        k = 0
        while self.register_next():
            k += 1
            if self.cfg.use_ba and k % self.cfg.ba_every == 0:
                self.bundle_adjust()
                self.filter_points()
        if self.cfg.use_ba:
            self.bundle_adjust()
        rem = self.filter_points()
        self.log["timing"]["total"] = time.time() - t0
        self.say(f"[done] {len(self.cams)}/{self.n} images registered, {len(self.points)} points "
                 f"(final filter removed {rem}) in {self.log['timing']['total']:.1f}s")
        return self

    # ------------------------------------------------------------------ results
    def point_cloud(self, with_colors=True):
        """Return dict(xyz, rgb, track_len, err_mean) for the final model."""
        pids = sorted(self.points)
        xyz = np.array([self.points[p] for p in pids]).reshape(-1, 3)
        tl = np.array([len(self.tracks[p]) for p in pids])
        errs = []
        for p in pids:
            e = [G.reproj_error(self.K, *self.cams[im], self.points[p][None], self.kps[im][kp][None])[0][0]
                 for im, kp in self.tracks[p].items() if im in self.cams]
            errs.append(np.mean(e))
        rgb = np.full((len(pids), 3), 180, np.uint8)
        if with_colors and pids:
            imgs = {}
            for k, p in enumerate(pids):
                im = max(self.tracks[p])  # colour from one observing image (cached)
                if im not in imgs:
                    imgs[im], _ = D.load_image(self.ds, self.names[im], self.cfg.max_dim)
                u, v = np.round(self.kps[im][self.tracks[p][im]]).astype(int)
                h, w = imgs[im].shape[:2]
                rgb[k] = imgs[im][np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)][::-1]
        return dict(pids=np.array(pids), xyz=xyz, rgb=rgb, track_len=tl, err_mean=np.array(errs))

    def observation_errors(self, min_track=2):
        """Per-observation reprojection errors (px). min_track=3 excludes 2-view points, whose error is
        near zero by construction (2 rays always nearly intersect) and therefore flatters the model."""
        out = []
        for pid, tr in self.tracks.items():
            if len(tr) < min_track:
                continue
            for im, kp in tr.items():
                if im in self.cams:
                    R, t = self.cams[im]
                    out.append(G.reproj_error(self.K, R, t, self.points[pid][None], self.kps[im][kp][None])[0][0])
        return np.array(out)

    def stats(self):
        pc = self.point_cloud(with_colors=False)
        oe = self.observation_errors()
        oe3 = self.observation_errors(min_track=3)
        inl = [m["n_inl"] for m in self.matches.values()]
        good = [m for m in self.matches.values() if m["n_inl"] >= self.cfg.min_pair_inliers]
        return dict(
            n_images=self.n, n_registered=len(self.cams), n_points=int(len(pc["xyz"])),
            mean_track_len=float(pc["track_len"].mean()) if len(pc["xyz"]) else 0.0,
            n_observations=int(len(oe)),
            reproj_mean=float(oe.mean()) if len(oe) else float("nan"),
            reproj_median=float(np.median(oe)) if len(oe) else float("nan"),
            reproj_rms=float(np.sqrt(np.mean(oe ** 2))) if len(oe) else float("nan"),
            n_points_track3=int((pc["track_len"] >= 3).sum()),
            reproj_mean_track3=float(oe3.mean()) if len(oe3) else float("nan"),
            reproj_median_track3=float(np.median(oe3)) if len(oe3) else float("nan"),
            mean_features=float(np.mean(self.log["n_features"])),
            n_pairs=len(self.matches), n_pairs_verified=len(good),
            mean_inliers_per_pair=float(np.mean(inl)) if inl else 0.0,
            mean_inlier_ratio=float(np.mean([m["n_inl"] / max(m["n_mutual"], 1) for m in good])) if good else 0.0,
            seconds=float(self.log["timing"].get("total", 0.0)),
            init_pair=[self.names[k] for k in self.init_pair] if self.init_pair else None,
        )
