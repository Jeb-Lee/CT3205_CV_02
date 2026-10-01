"""South-building: classical vs GPU vs learned (DISK + LightGlue) on the control sequence and the viewpoint ladder.

    python experiments/exp_learned.py
Writes results/expL/results.csv|md. Same control as Experiments A-D (12 images, stride 2) unless stated.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))
from sfm import datasets as D
from sfm.reconstruction import SfMConfig
from run_experiments import run_one, save_table, COLS, START

ds = D.load_dataset("south-building")
out = os.path.join("results", "expL"); os.makedirs(out, exist_ok=True)
rows = []
cls = dict(detector="rootsift", matcher="bf"); gpu = dict(detector="rootsift", matcher="gpu"); dl = dict(detector="disk", matcher="lightglue")
names = D.select_images(ds, START, 12, 2)
for lab, kw in (("control: RootSIFT+BF (CPU)", cls), ("RootSIFT+matcher on GPU", gpu), ("DISK+LightGlue (GPU)", dl)):
    rows.append(run_one(ds, lab, names, SfMConfig(**kw), out_dir=out) | dict(group="control 12 imgs", matcher=kw["matcher"]))
for stride in (3, 4, 5, 6):                              # the ladder where classical matching collapsed (Exp. C)
    names = D.select_images(ds, START, 8, stride)
    for lab, kw in ((f"stride={stride} RootSIFT", cls), (f"stride={stride} DISK+LG", dl)):
        rows.append(run_one(ds, lab, names, SfMConfig(**kw), out_dir=out) | dict(group="ladder 8 imgs", matcher=kw["matcher"]))
for n in (3, 4):                                         # the fragile few-image cases of Exp. A
    names = D.select_images(ds, START, n, 2)
    for lab, kw in ((f"N={n} RootSIFT", cls), (f"N={n} DISK+LG", dl)):
        rows.append(run_one(ds, lab, names, SfMConfig(**kw), out_dir=out) | dict(group="few images", matcher=kw["matcher"]))
save_table(rows, out, ["group"] + COLS)
