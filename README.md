# CT3205 Assignment 2 - Multi-View 3D Reconstruction and Point Cloud Generation

An incremental Structure-from-Motion pipeline implemented from the geometry up (OpenCV is used for feature
extraction, RANSAC and PnP; the essential-matrix decomposition, cheirality test, DLT triangulation, track management,
bundle adjustment and evaluation are our own code).

## Install
```bash
python -m pip install -r requirements.txt      # Python 3.10-3.13; developed on 3.13
python scripts/check_env.py                    # prints the versions in use
```

## Data
`Datasets/south-building` (main), `Datasets/person-hall` (challenging scenario): download from
<https://demuc.de/colmap/datasets/> (images + COLMAP sparse model in text format). The COLMAP model is used only for intrinsics
and as ground truth for evaluation. Own photos: put them in `Datasets/<name>/images/` (optional `intrinsics.json`).

## Reproduce
```bash
# 1. baseline reconstruction + all report figures (about 1 min)
python scripts/run_baseline.py --start 40 --count 12 --stride 2 --out results/baseline --video

# 2. experiments A (number of images), B (detector), C (viewpoint), D (image quality)
python experiments/run_experiments.py A B C D

# 3. dense stereo reconstruction (advanced work): rectified SGBM + multi-pair fusion, with ablation and video
python scripts/run_dense.py --start 40 --count 16 --stride 1 --out results/dense --ablate --video

# 4. challenging scenarios on Person Hall (cases A-F3r), per-case figures + cases.csv
python scripts/run_person_hall.py --cases A B C D E2 E4 E6 E8 F3 --video
python scripts/diag_case.py C                 # per-camera / per-pair diagnosis against COLMAP GT

# 5. GPU / learned features (needs the CUDA build of PyTorch, see requirements.txt) and COLMAP comparison
python scripts/gpu_bench.py                   # GPU matcher == CPU matcher? speed; DISK + LightGlue sanity
python scripts/matcher_bakeoff.py             # SIFT vs DISK vs DISK+LightGlue on hard pairs, scored with COLMAP geometry
python experiments/exp_learned.py             # south-building: classical vs GPU vs learned
python scripts/run_person_hall.py --cases B_dl D_dl F_dl F3_dl B_dls D_dls F_dls F3_dls --video
python scripts/diag_run.py --dataset person-hall --start 0 --count 110 --stride 3 --detector disk --matcher lightglue --window 6 --ba_every 10 --min_pair 150 --tag F3_dls
python scripts/run_colmap_compare.py          # ours vs COLMAP on identical images

# 6. own captures (iPhone photos in Datasets/Bottle, Fire est, Snacks, yellow_sign): QA, self-calibration, reconstruction, dense
python scripts/check_capture.py --matching        # pre-flight: EXIF, sharpness, neighbour overlap, contact sheets
python scripts/estimate_focal.py --set Snacks     # focal length from fundamental matrices (validated on south-building)
sh scripts/_run_own_chain.sh                      # reconstruct all five sets (about 1.5 h; all-pairs matching)
python scripts/redo_dense.py --tag yellow_all --gap 3 --crop 0.3 --axial 1.0      # object-focused dense stage
python scripts/orbit_check.py; python scripts/render_own_figures.py; python scripts/render_own_turntables.py

# 7. interactive viewer (screenshots for the report) and the auto-generated demo clip
python scripts/view_open3d.py results/baseline --gt
python scripts/view_open3d.py results/dense --ply dense.ply   # dense cloud
python scripts/export_html.py            # self-contained interactive HTML viewer -> results/viewer/south_building.html
python scripts/make_demo_video.py --out results/demo
```
Model weights: the classical pipeline has no learned parameters. The advanced learned-feature work uses two **pretrained**
models (no training by us), copies in `weights/`: DISK (`disk_depth-save.pth`, from github.com/cvlab-epfl/disk) and LightGlue for
DISK (`lightglue_disk_v0-1_arxiv.pth`, from github.com/cvg/LightGlue releases). kornia downloads them automatically on first use
(to `~/.cache/torch/hub/checkpoints`). Check the upstream licences before redistributing.

## Layout
| Path | Content |
|---|---|
| `sfm/features.py`, `matching.py` | SIFT/RootSIFT/ORB/AKAZE/BRISK, ratio + mutual + RANSAC-E filtering |
| `sfm/geometry.py` | E decomposition (4 solutions), cheirality, DLT triangulation, reprojection, parallax |
| `sfm/reconstruction.py` | incremental SfM: init pair, PnP registration, triangulation, tracks, filtering |
| `sfm/gpu_matching.py` | GPU brute-force matcher, DISK + LightGlue (learned) |
| `sfm/colmap_compare.py` | ours vs COLMAP on identical images |
| `sfm/dense.py` | dense stereo: rectification, SGBM, voxel-vote fusion |
| `sfm/bundle.py` | sparse bundle adjustment (SciPy, Huber loss) |
| `sfm/evaluate.py` | Umeyama alignment to COLMAP, pose / point / completeness metrics |
| `sfm/viz.py` | match figures, cloud + camera frusta, reprojection overlays, PLY, turntable |
| `scripts/` | baseline runner, diagnostics, Open3D viewer, demo video |
| `experiments/run_experiments.py` | Experiments A-D |
| `docs/PROGRESS.md` | detailed development log; `docs/GUIDE_YOUR_TODO.md` capture guide and to-do |

## AI usage
See the AI Usage Declaration in the report (AI-assisted code generation and review; all outputs verified by the author).
