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

# 4. interactive viewer (screenshots for the report) and the auto-generated demo clip
python scripts/view_open3d.py results/baseline --gt
python scripts/view_open3d.py results/dense --ply dense.ply   # dense cloud
python scripts/export_html.py            # self-contained interactive HTML viewer -> results/viewer/south_building.html
python scripts/make_demo_video.py --out results/demo
```
Trained model weights: none - the classical pipeline has no learned parameters. (Advanced work using learned features will
document the pretrained weights used.)

## Layout
| Path | Content |
|---|---|
| `sfm/features.py`, `matching.py` | SIFT/RootSIFT/ORB/AKAZE/BRISK, ratio + mutual + RANSAC-E filtering |
| `sfm/geometry.py` | E decomposition (4 solutions), cheirality, DLT triangulation, reprojection, parallax |
| `sfm/reconstruction.py` | incremental SfM: init pair, PnP registration, triangulation, tracks, filtering |
| `sfm/dense.py` | dense stereo: rectification, SGBM, voxel-vote fusion |
| `sfm/bundle.py` | sparse bundle adjustment (SciPy, Huber loss) |
| `sfm/evaluate.py` | Umeyama alignment to COLMAP, pose / point / completeness metrics |
| `sfm/viz.py` | match figures, cloud + camera frusta, reprojection overlays, PLY, turntable |
| `scripts/` | baseline runner, diagnostics, Open3D viewer, demo video |
| `experiments/run_experiments.py` | Experiments A-D |
| `docs/PROGRESS.md` | detailed development log; `docs/GUIDE_YOUR_TODO.md` capture guide and to-do |

## AI usage
See the AI Usage Declaration in the report (AI-assisted code generation and review; all outputs verified by the author).
