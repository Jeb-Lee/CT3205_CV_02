# Progress log - CT3205 Assignment 2 (multi-view 3D reconstruction)

Living lab notebook. Each entry: what was done, why, numbers, and what it means for the report.
Numbers are copied from `results/**/stats.json` / `results.csv`; regenerate them with the commands shown.

## 0. Environment (2026-09-30)
- Python 3.13, OpenCV 5.0.0.93, NumPy 2.4.4, SciPy 1.18.1, Matplotlib 3.10.8, Open3D 0.20.0 (`requirements.txt`, `scripts/check_env.py`).
- GPU: RTX 3050 6 GB (torch installed is CPU-only; CUDA build only needed for the advanced learned-feature work).
- Datasets: `Datasets/south-building` (128 images, 3072x2304, SIMPLE_RADIAL f=2559.7, k1=-0.0205), `Datasets/person-hall` (330 images, 5616x3744, OPENCV model). COLMAP `sparse/` folders are used only for intrinsics and **ground-truth evaluation**, never as input to the reconstruction.

## 1. Design decisions
| Decision | Choice | Reason |
|---|---|---|
| Main dataset | south-building | Manageable size, COLMAP ground truth for quantitative error |
| Pipeline | Own incremental SfM (`sfm/`) | Brief requires understanding, not just running COLMAP; COLMAP is used only for comparison (advanced part) |
| Working resolution | longest side 1600 px, K scaled accordingly | Speed; distortion removed by `cv2.undistort` so the rest is a pure pinhole model |
| Detector | RootSIFT, 8000 features (SIFT/ORB/AKAZE/BRISK compared in Exp. B) | RootSIFT is a free improvement over SIFT descriptors |
| Match filtering | Lowe ratio 0.8 -> mutual check -> RANSAC on E (1 px) | The cascade shown in the report figures |
| Two-view pose | own SVD decomposition of E + cheirality on 4 candidates; cross-checked with `cv2.recoverPose` | Brief 1.3 wants the ambiguity explained |
| Triangulation | own DLT in normalised coordinates (batched SVD); cross-checked with `cv2.triangulatePoints` (max diff 8e-12) | Brief 1.5 |
| Multi-view | Init pair -> PnP registration -> triangulate -> sparse BA (scipy) | "Most important part" of the report |
| Scale | Pair-1 baseline = 1 (monocular ambiguity) | Comparison to COLMAP uses Umeyama similarity alignment |

## 2. Implementation log
1. **Data layer** (`sfm/io_colmap.py`, `sfm/datasets.py`): COLMAP text readers, image loading with resize + undistort, controlled degradations (blur/noise/contrast/gamma) for Exp. D, custom-dataset support (EXIF focal length fallback) for own photos.
2. **Features/matching** (`sfm/features.py`, `sfm/matching.py`).
3. **Geometry** (`sfm/geometry.py`): `decompose_essential`, `triangulate_dlt`, `recover_pose_cheirality`, projection/reprojection error, parallax angle, Sampson error.
4. **Reconstruction** (`sfm/reconstruction.py`): tracks stored as `pid -> {image: keypoint}`; new points must pass positive depth, reprojection < 2 px and parallax > 1.5 deg.
5. **Bundle adjustment** (`sfm/bundle.py`): Huber loss, analytic sparsity pattern, first camera fixed.
6. **Evaluation** (`sfm/evaluate.py`): RANSAC-Umeyama alignment to COLMAP; camera position/rotation error, point accuracy, completeness; GT-epipolar labelling of matches as correct/incorrect (for the "correct vs incorrect matches" figures).
7. **Visualisation** (`sfm/viz.py`): match stages, scene with frusta + viewing directions, reprojection overlay + histogram, PLY export, turntable video.

## 3. Bugs / observations along the way (worth a sentence in the report)
- **Noise pairs leaking into registration.** With 8-point RANSAC, unrelated image pairs still report ~8-10 "inliers". They were being used as if valid matches. Fix: pairs with `< 25` verified inliers are ignored (`min_pair_use`).
- **First run (stride 2 from image 0): only 8/16 images registered.** The inlier matrix (`scripts/diag_pairs.py`) showed a strongly connected cluster (2000+ inliers) and a weakly connected tail (30-130 inliers) where the camera goes around the building corner and foliage dominates. Incremental SfM cannot bridge a chain with < ~25 2D-3D correspondences. This is a real failure mode, kept for the "challenging scenarios" section. The main baseline therefore uses images 181-203 (start=40, stride 2), which are richly connected.
- **Reprojection error can flatter the model.** 2-view tracks fit their own two observations almost exactly (median 0.07 px). I therefore also report the error for tracks with >= 3 views (`reproj_mean_track3`) - the honest number.
- **Display outliers.** Far, low-parallax points (distant trees) stretched the plot axes; the plots now hide points > 4x median distance (the PLY keeps them).

## 4. Baseline result (12 images: P1180181...P1180203, stride 2) - `results/baseline/`
Command: `python scripts/run_baseline.py --start 40 --count 12 --stride 2 --out results/baseline --video`

| Metric | Value |
|---|---|
| Images registered | 12 / 12 |
| 3D points | 10,934 (4,986 seen in >= 3 views) |
| Mean track length | 2.95 |
| Reprojection error mean / median | 0.138 / 0.103 px (>=3-view tracks: 0.158 / 0.124) |
| Init pair | P1180189 + P1180191, median parallax 18.5 deg, cheirality counts [0, 3077, 0, 0] |
| Camera position error vs COLMAP | 0.11 % of trajectory extent (RMSE) |
| Camera rotation error | mean 0.18 deg, max 0.37 deg |
| Points within 2 % of extent of a COLMAP point | 99.4 % |
| Completeness (COLMAP points with one of ours nearby) | 36 % (ours is sparser by design: 8000 features/image) |
| Runtime | 66 s |

Figures: `scene_views.png`, `matches/*.jpg` (raw / ratio+mutual / RANSAC, red = wrong under GT geometry), `reproj_*.jpg`, `reproj_hist.png`, `two_view_demo.json` (four candidate poses, only one has points in front of both cameras), `turntable.mp4`.

## 5. Experiments A-D (in progress)
Runner: `experiments/run_experiments.py A B C D` -> `results/exp{A,B,C,D}/`. Control = baseline configuration; one variable changed per experiment.
- **A** number of images: 3,4,6,8,12,16,20 (nested subsets, stride 2).
- **B** detector: SIFT, RootSIFT, ORB, AKAZE, BRISK on the same 12 images.
- **C** viewpoint change: 8 images with stride 1..6, plus a two-view study of depth quality vs baseline.
- **D** image quality: blur (sigma 1,2,4,8), noise (10,25,50), resolution (1200,800,400,200 px), contrast (x0.5, 0.25, 0.1), gamma (2.0, 3.5).

Hypotheses (to be confirmed or refuted by the data - write the outcome next to each in the report):
- A: more images -> more points and lower pose error with diminishing returns; 3 images may fail or be fragile; runtime grows about quadratically (all-pairs matching).
- B: SIFT-family >> binary descriptors on wide baselines; ORB gives most points per second but lowest inlier ratio.
- C: stride 1 has tiny parallax -> noisy depth; very large stride breaks matching; an optimum exists in between.
- D: blur and low resolution kill features earliest; noise hurts ORB more than SIFT; a global contrast change should hurt little (descriptors are contrast-normalised).

(Results filled in below once the runs finish.)
