"""Post-hoc focal-length refinement: start from a CONVERGED model (many cameras, varied depth) and let bundle adjustment move
one shared focal length. Reports the focal change and the reprojection error before / after.

    python scripts/refine_intrinsics.py --tag bottle_classic
"""
import argparse, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm.bundle import bundle_adjust

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--nfev", type=int, default=60)
a = ap.parse_args()
d = os.path.join("results", "own", a.tag)
rec = pickle.load(open(os.path.join(d, "rec.pkl"), "rb"))
e0 = rec.observation_errors(); f0 = float(rec.K[0, 0])
t0 = time.time()
cams, pts, (r0, r1), fs = bundle_adjust(rec.K, rec.cams, rec.points, rec._observations(), fixed=rec.init_pair[0], max_nfev=a.nfev, refine_f=True)
rec.cams, rec.points = cams, pts
rec.K = rec.K.copy(); rec.K[0, 0] *= fs; rec.K[1, 1] *= fs
e1 = rec.observation_errors()
out = dict(tag=a.tag, f_before=f0, f_after=float(rec.K[0, 0]), f_change_pct=100 * (fs - 1), rmse_before=r0, rmse_after=r1,
           mean_reproj_before=float(e0.mean()), mean_reproj_after=float(e1.mean()), median_before=float(np.median(e0)), median_after=float(np.median(e1)),
           n_cams=len(cams), n_points=len(pts), seconds=time.time() - t0)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
json.dump(out, open(os.path.join(d, "refine_intrinsics.json"), "w"), indent=1)
