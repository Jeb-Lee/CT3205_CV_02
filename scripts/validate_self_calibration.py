"""Does bundle-adjustment focal refinement recover the true focal length? (south-building has COLMAP's calibrated value)

    python scripts/validate_self_calibration.py
Starts from a focal length that is wrong by -12 %, 0 % and +12 % and reports the refined value, reprojection error and
camera error against COLMAP. Writes results/own/self_calibration_validation.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import datasets as D, evaluate as Ev
from sfm.reconstruction import Reconstruction, SfMConfig

ds = D.load_dataset("south-building")
names = D.select_images(ds, 40, 12, 2)
f_true = float(ds.K[0, 0]) * 1600 / max(ds.size)
rows = []
for fac in (0.88, 1.0, 1.12):
    ds.K[0, 0] = ds.K[1, 1] = 2559.68 * fac
    for refine in (False, True):
        rec = Reconstruction(ds, names, SfMConfig(matcher="gpu", refine_focal=refine), cache_dir="results/cache", verbose=False).run()
        st = rec.stats(); gt = Ev.evaluate_against_gt(rec); gt.pop("_sim3", None)
        r = dict(start_factor=fac, refine=refine, f_start=2559.68 * fac * 1600 / max(ds.size), f_final=float(rec.K[0, 0]), f_true=f_true,
                 f_error_pct=100 * (float(rec.K[0, 0]) / f_true - 1), registered=st["n_registered"], points=st["n_points"],
                 reproj=st["reproj_mean"], cam_pos_rel_pct=100 * gt.get("cam_pos_rmse_rel", np.nan), cam_rot_deg=gt.get("cam_rot_mean_deg", np.nan))
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(rows, open("results/own/self_calibration_validation.json", "w"), indent=1, default=float)
