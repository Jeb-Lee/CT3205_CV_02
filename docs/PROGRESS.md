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

## 5. Experiments A-D (design)
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

(Outcomes: section 6.)

## 6. Experiment results (clean sequential run; commands: `python experiments/run_experiments.py A B C D`)
Control: 12 images P1180181..203 (stride 2), RootSIFT 8000 features, 1600 px, ratio 0.8 + mutual + RANSAC-E 1 px, PnP+BA. Data: `results/exp{A,B,C,D}/results.csv|md` and `*_metrics.png`.
"pts>=3" = points seen in >= 3 views. "acc" = share of our points within 2 % of the trajectory extent of a COLMAP point after Sim(3) alignment.

### Experiment A - number of images (nested subsets, stride 2)
| N | registered | 3D pts | pts>=3 views | cam pos err (rel.) | rot err (deg) | mean reproj (px) | completeness vs COLMAP | time (s) |
|---|---|---|---|---|---|---|---|---|
| 3 | 3/3 | 1,448 | **21** | 0.0017 | 0.15 | 0.112 | 4 % | 1.5 |
| 4 | **3/4** | 2,232 | 70 | 0.0062 | 0.87 | 0.105 | (unreliable, see below) | 3.7 |
| 6 | 6/6 | 6,589 | 2,237 | 0.0013 | 0.23 | 0.113 | 19 % | 14 |
| 8 | 8/8 | 8,889 | 4,097 | **0.0008** | 0.14 | 0.129 | 23 % | 28 |
| 12 | 12/12 | 10,934 | 4,986 | 0.0011 | 0.18 | 0.138 | 36 % | 62 |
| 16 | 16/16 | 18,849 | 9,193 | 0.0041 | 0.37 | 0.172 | 54 % | 139 |
| 20 | 20/20 | 23,294 | 12,375 | 0.0029 | 0.30 | 0.181 | 63 % | 212 |

Findings
- Points and completeness grow steadily with N (1.4k -> 23k points; completeness 4 % -> 63 %); runtime grows roughly quadratically (all-pairs matching): 212 s at N=20.
- **More images does NOT always improve accuracy**: camera error is best at N=8 (0.0008) and *worse* at N=16/20 (0.004 / 0.003). Untested hypotheses: the later images sit in weaker-overlap / foliage-heavy parts of the sequence, and more chained registrations accumulate drift.
- **Mean reprojection error rises with N (0.11 -> 0.18 px)** although the model gets better: with N=3 nearly all points are 2-view and 2-view points fit almost perfectly by construction (only 21 points have >= 3 views). Reprojection error must be read together with track length.
- **N=4 did not use all its images**: image 181 links to the initial pair (185,187) with only 9-10 "inliers" (RANSAC noise) so it is correctly rejected; 183 joins with only 71 2D-3D correspondences. Its accuracy score (0.9 % of points within 2 %) is *misleading*: `scripts/diag_alignment.py` shows that with a camera-independent scaled-ICP alignment 78-80 % of the points are within 2 %, i.e. the 3-camera Sim(3) alignment, not the points, is the problem. Camera error after ICP is still 6-8 % of the extent, so it is a fragile reconstruction. Lesson: with 3-4 nearly collinear cameras both the reconstruction and its evaluation are ill-conditioned. (An optical-axis alignment fix for near-collinear cameras was added to `evaluate.py`.)

### Experiment B - detector / descriptor (same 12 images)
| detector | registered | 3D pts | verified inliers / pair | inlier ratio | mean reproj (px) | cam err (rel.) | time (s) |
|---|---|---|---|---|---|---|---|
| SIFT | 11/12 | 9,414 | 452 | 0.80 | 0.129 | 0.0012 | 56 |
| **RootSIFT** | **12/12** | **10,934** | **483** | **0.86** | 0.138 | 0.0011 | 57 |
| ORB | 8/12 | 3,706 | 115 | 0.61 | 0.351 | 0.0123 | 29 |
| AKAZE | 8/12 | 5,112 | 201 | 0.66 | 0.222 | 0.0020 | 37 |
| BRISK | 9/12 | 6,642 | 271 | 0.79 | 0.209 | 0.0013 | 48 |

Findings: the SIFT family clearly wins on this wide-baseline outdoor sequence; RootSIFT (a free L1 + sqrt normalisation) beats plain SIFT in inliers, inlier ratio, points and registration. ORB is ~2x faster but has 4x fewer inliers, ~3x higher reprojection error and ~10x worse camera error. The binary detectors were not tuned (near-default thresholds, shared ratio 0.8), so their ranking is "out of the box"; a fair-tuning study is a possible extension. (A first attempt crashed: OpenCV 5 moved AKAZE/BRISK to `cv2.xfeatures2d`; fixed in `sfm/features.py`.)

### Experiment C - viewpoint change (8 images, stride 1..6)
| stride | registered | verified inliers / pair | 3D pts | acc (within 2 %) | cam err (rel.) |
|---|---|---|---|---|---|
| 1 | 8/8 | 988 | 10,833 | 0.917 | 0.0024 |
| 2 | 8/8 | 866 | 8,889 | 0.992 | 0.0008 |
| 3 | 6/8 | 358 | 4,985 | 0.995 | 0.0029 |
| 4 | **2/8** | 265 | 2,189 | n/a | n/a |
| 5 | 3/8 | 224 | 2,982 | 0.989 | 0.0020 |
| 6 | 3/8 | 140 | 2,420 | 0.992 | 0.0030 |

Two-view study (`results/expC/two_view_baseline_study.csv`, frame i vs i+k): k=1: 2,520 inliers, median parallax 12.6 deg; k=2: 1,270 / 24 deg; k=3: 377 / 34 deg; **k>=4: 9-10 inliers, i.e. matching collapses** (roughly 35+ deg viewpoint change on this facade).
Findings: matches and points fall monotonically with viewpoint change, registration collapses beyond stride 3, and stride 1-2 is the sweet spot. Point accuracy is worst at stride 1 (0.917 vs 0.992): smallest baseline -> weakest depth, consistent with theory. **Caveat:** even consecutive south-building frames already have ~12 deg parallax, so this dataset never reaches the true "insufficient camera movement" regime; that case needs own photos with tiny hand-held steps (see GUIDE_YOUR_TODO.md).

### Experiment D - image quality (baseline 12/12 registered, 10,934 pts)
| factor | level | registered | 3D pts | cam err (rel.) |
|---|---|---|---|---|
| blur sigma (px) | 1 / 2 / 4 / 8 | 12 / 12 / 8 / 2 | 11,144 / 6,872 / 740 / 41 | .0020 / .0025 / .0038 / n/a |
| noise sigma (grey levels) | 10 / 25 / 50 | 11 / 8 / 2 | 7,070 / 1,505 / 61 | .0027 / .0024 / n/a |
| resolution (longest side) | 1200 / 800 / 400 / 200 | 12 / 12 / 8 / 6 | 8,986 / 5,161 / 851 / 166 | .0012 / .0018 / .0061 / .0152 |
| contrast | x0.5 / x0.25 / x0.1 | 12 / 11 / **0 (fail)** | 10,800 / 5,846 / 0 | .0018 / .0018 / n/a |
| gamma (darker) | 2.0 / 3.5 | 12 / 12 | 12,366 / 12,440 | .0011 / .0015 |

Findings: blur, noise and resolution degrade *gracefully, then collapse*: points fall first (blur s=2 halves them, s=4 leaves 7 %), then registration, then the initial pair itself. Blur is the most destructive per unit; blur sigma >= 4 or noise sigma 50 leaves too few features to build a model. Global contrast reduction is tolerated down to x0.25 and fails at x0.1 (8-bit quantisation destroys the weak gradients before descriptors can use them). Darkening via gamma does not hurt (even slightly more points): descriptors are gradient-normalised. Camera error stays around 0.1-0.3 % of the extent for everything that still registers, so degradation shows up mainly as **lost points and images**, not as pose drift. Reprojection error of failing runs is low only because few points survive (survivorship); do not read it as "better".

Caveat for all experiments: one sequence, and RANSAC randomness is not averaged (the runs reproduced identically on re-execution in this environment).
