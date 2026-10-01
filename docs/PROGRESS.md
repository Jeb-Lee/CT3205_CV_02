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

## 7. Advanced work: dense stereo reconstruction (2026-10-01)
Code: `sfm/dense.py`, runner `scripts/run_dense.py`. Command: `python scripts/run_dense.py --start 40 --count 16 --stride 1 --out results/dense --ablate --video`.

**Method.** (1) SfM on 16 *consecutive* images (P1180181-196); consecutive frames are used because dense matching needs modest convergence (about 10-12 deg), unlike sparse SfM which tolerates ~35 deg. (2) For every neighbouring pair of *estimated* cameras: `cv2.stereoRectify` (Bouguet) so epipolar lines become image rows, then SGBM (3-way, block 5, uniqueness 10, left-right check 1) with the disparity search range derived from the sparse points seen in that image (2nd-98th percentile depth +/- 25 %). (3) `reprojectImageTo3D` with Q -> rectified frame -> camera frame -> world. (4) **Multi-view fusion by voxel voting**: a voxel (radius/400) is kept only if >= 2 different stereo pairs put a point in it; kept points are averaged. Fusion is what removes SGBM mismatches: single-pair results are noisy, agreement between independent pairs is not.

**Sparse stage on this sequence:** 16/16 registered, 18,216 points, camera position error 0.05 % of trajectory extent, rotation error 0.068 deg (mean). Denser, consecutive views give a *better* sparse model than the stride-2 baseline (0.11 %).

**Result:** 15 stereo pairs, 3.40 M raw points -> **428,018 fused points = 23.5x the sparse cloud**, fused in 18 s (all stereo pairs together < 10 s on CPU). Figures: `results/dense/sparse_vs_dense.png` (side by side), `stereo_pair.jpg` (rectified pair with epipolar rows - window rows line up - and the disparity map), `dense_turntable.mp4`, `dense.ply`.

| metric (vs COLMAP ground truth after Sim(3) alignment) | sparse | dense |
|---|---|---|
| points | 18,216 | 428,018 |
| COLMAP points (inside our reconstructed volume, n = 22,141) with one of our points within 2 % of scene radius | 58.5 % | **66.4 %** |
| median gap COLMAP point -> nearest of our points (fraction of scene radius) | 0.0138 | **0.0046** |
| median distance of our points to nearest COLMAP point (fraction of scene radius) | 0.0075 | 0.0101 |
| dominant-plane (facade) fit: inlier share / RMS (fraction of camera-arc extent) | 70 % / 0.0024 | 67 % / 0.0032 |

Reading: the dense cloud fills in ~3x more of the surface (gap 0.014 -> 0.005) with 23x more points, while individual points are slightly *less* precise than sparse SfM points (RMS +35 %, distance to COLMAP +35 %): sparse points are sub-pixel-refined feature points, dense points are per-pixel SGBM estimates (SGBM disparity resolution 1/16 px, no per-point refinement). The COLMAP reference is itself sparse (61 k points) so "distance to COLMAP" is a loose accuracy check; the plane residual is the tighter one.

### Ablation (one SfM run, dense settings varied; `results/dense/ablation.csv`)
| config | pairs | fused pts | COLMAP within 2 % (scene) | dist to COLMAP | facade-plane RMS | plane inlier share |
|---|---|---|---|---|---|---|
| block 3 | 15 | 484,249 | 66.5 % | 0.0099 | 0.0032 | 70 % |
| block 5 (default) | 15 | 428,018 | 66.4 % | 0.0101 | 0.0032 | 67 % |
| block 9 | 15 | 378,165 | 66.1 % | 0.0102 | 0.0035 | 65 % |
| pair gap 2 (wider baseline) | 13 | 169,492 | 69.5 % | 0.0142 | 0.0032 | **92 %** |
| pair gap 3 | 10 | **76** | 0.1 % | 0.0173 | - | - |
| min support 1 (no fusion check) | 15 | 1,105,316 | **74.9 %** | 0.0133 | 0.0038 | **46 %** |
| min support 3 | 15 | 224,373 | 66.5 % | 0.0109 | **0.0030** | 80 % |

Findings
- **Fusion support is the main quality knob**: support 1 gives 2.6x more points and highest coverage but the noisiest (only 46 % of points on the facade plane); support 3 halves the points but is cleanest (80 %, lowest RMS). A completeness/noise trade-off.
- **Block size** matters little: 3 gives ~13 % more points and marginally cleaner planes; 9 smooths more and loses fine detail.
- **Wider baseline (gap 2)** gives much flatter facades (92 % plane inliers vs 67 %): triangulation with a larger angle is better conditioned - the same geometry as Experiment C - but fewer points survive because overlap shrinks.
- **Gap 3 collapses (76 points)**: too much convergence for rectified SGBM, matching the sparse-stage limit found in Experiment C (~35 deg).
- Single-pair quality varies enormously (valid pixels 1 %-19 % of the image, maximum 25 % because of the 2-pixel step): pairs where the disparity range is huge or half of the image has no overlap are poor, and only multi-pair fusion makes a clean cloud.

Limitations to state in the report: dense matching is 2-view SGBM (no photometric multi-view consistency, no depth-map refinement), so foliage, thin structures, windows/glass and the sky are unreliable or absent; points near the edges of pairs lack support and are dropped; the accuracy reference (COLMAP sparse) is loose. Not implemented: surface reconstruction (Poisson), a GPU plane-sweep (planned comparison for the RTX 3050).

## 8. Challenging scenarios: Person Hall (2026-10-01)
Code: `scripts/run_person_hall.py` (cases), `scripts/diag_case.py` (per-camera / per-pair diagnosis against COLMAP GT), `scripts/montage.py`.
Commands: `python scripts/run_person_hall.py --cases A B C D E2 E4 E6 E8 F3 F --video`; results in `results/person_hall/<case>/` and `results/person_hall/cases.csv`.
Dataset: 330 photos of a brick building walked fully around (closed loop), 5616x3744, OPENCV camera model with strong distortion (k1 = -0.11) -> contact sheet `results/person_hall/contact_sheet.jpg`. Working resolution 1600 px, default pipeline (RootSIFT 8000, PnP + BA), nothing tuned per case.
Outcome rule fixed **before** looking at results (`classify()`): SUCCESS = >= 90 % of images registered and camera position error < 2 % of the trajectory extent; PARTIAL = >= 50 % registered; FAILURE otherwise.

### 8.1 Results of all cases (final, fixed loader)
| case | what it tests | images | registered | points | camera pos err | rot err (deg) | outcome |
|---|---|---|---|---|---|---|---|
| A | clean brick facade + obelisk, stride 3 | 12 | 12 | 12,584 | 0.03 % | 0.03 | **SUCCESS** |
| B | foliage occlusion (frames 1045-1078) | 12 | **2** | 63 | n/a | n/a | **FAILURE** |
| C | repetitive arched windows + sun flare | 12 | 12 | 6,509 | 0.11 % | 0.15 | **SUCCESS** (after bug fix, 8.2) |
| D | corner turn, big viewpoint + scale change | 12 | 11 | 5,226 | 0.07 % | 0.10 | **SUCCESS** (IMG_1313 not registered) |
| E2 / E4 / E6 / E8 | viewpoint sweep on the clean facade, stride 2 / 4 / 6 / 8 | 8 each | 8 / 8 / 8 / 8 | 10,351 / 8,764 / 7,067 / 6,487 | 0.04 / 0.04 / 0.05 / **0.75** % | 0.02 / 0.09 / 0.05 / **0.99** | SUCCESS x4 (accuracy degrades at stride 8) |
| F | whole loop, every 6th image, window 8 | 55 | **15** | 10,216 | 0.66 % | 0.56 | **FAILURE** |
| F3 | whole loop, every 3rd image, window 6 | 110 | **44** | 24,143 | 0.16 % | 0.18 | **FAILURE** (partial: one full side of the building) |
| F3r | as F3 with relaxed thresholds (ratio 0.85, pair >= 15 inliers, PnP >= 12) | 110 | **45** | 25,125 | 0.19 % | 0.26 | **FAILURE** |

Inliers per verified pair in the sweep: E2 1,774 -> E4 734 -> E6 441 -> E8 466 (verified pairs 28 -> 22 -> 17 -> 17): matches fall quickly with viewpoint change, yet registration still succeeds on this facade, unlike the much harder south-building stride 4 (Exp. C). Stride is **not comparable across datasets**; parallax between views is the real variable.

### 8.2 The big one: a "successful" wrong reconstruction, caused by a bug of mine
First run of case C: **12/12 images registered, reprojection error only 0.6 px, 6.5k points - but camera position error 51 % of the trajectory and rotation error 139 deg.** Everything an automatic check looks at said "fine"; only ground truth showed it was wrong. Figure `results/person_hall/exif_bug_before_after.png`.
Diagnosis (`scripts/diag_case.py C`, figures in `results/person_hall/_exif_bug_before/C/`):
1. 50 of 60 verified pairs had ~100 % of their matches "violating" the GT epipolar geometry, even for adjacent frames, yet the matches looked visually right (windows linked to the same windows).
2. GT relative rotation IMG_1228 -> IMG_1230 is **91 deg** between near-identical views: the two images are stored differently.
3. `PIL` showed **48 of 330 photos (indices 215-242, 254-273) carry EXIF orientation 8 (portrait)**. `cv2.imread` silently rotates them upright; COLMAP ignores the tag (treats every image as a landscape sensor image with one K). My pipeline therefore fed rotated images with an un-rotated K, and compared against a GT in another convention.
4. Fix: `cv2.IMREAD_IGNORE_ORIENTATION` in `sfm/datasets.py`; feature cache key bumped. Result: C camera error 51 % -> **0.11 %**, D 0.97 % (rot 9.8 deg) -> **0.07 %** (rot 0.10 deg). South-building has no tagged images (0 of 128), and A, B, E involve none, so their earlier results are unchanged (re-run: identical numbers).
Lessons for the report: (a) a reconstruction can be internally consistent (low reprojection error, all images registered) and globally wrong - validate with independent evidence (GT, or at least pairwise geometric consistency); (b) loader conventions must match the camera model; (c) it was caught because every failure was investigated instead of reported as "repetitive structure made it fail". **For your own photos keep the phone in landscape** and check `PIL.Image.getexif()` orientation tags.

### 8.3 Case B - foliage occlusion (real failure)
2 of 12 registered, 63 points, only 6 verified pairs (mean 16 inliers per pair, close to the RANSAC noise floor of ~8-10). Figure `results/person_hall/B/matches/*_c_ransac.jpg`: a conifer stands in front of the wall in the second image; the **74 matches that survive are all correct (0 wrong against GT)** but far too few and confined to the visible part of the wall. Failure mode = **occlusion / view-dependent foliage texture, not false matches**: matching finds only a small consistent subset, the pair graph falls apart (`B/inlier_matrix.png`), and there is nothing for PnP to attach new images to. Fixing it would need much denser sampling, or features on the building only (semantic masking of vegetation).

### 8.4 Case D - which image is lost and why
IMG_1313 is the only image not registered (`D/unregistered_context.jpg`): the photographer stepped back, so this frame shows the building corner from about **twice the distance** between two close-ups of the same wall (IMG_1309, IMG_1317). Likely cause (not measured): the large scale + viewpoint change leaves too few matches with already-tracked points for PnP; SIFT is scale-invariant only within a limited range.

### 8.5 Whole-building loop (F, F3, F3r): failure with a measured cause
The full loop has 330 photos; all-pairs matching is 54,285 pairs (about 6.5 h at the measured 0.43 s / pair), so windowed matching was used. Measured on the pair graph (edge = verified pair with >= 25 inliers):

| run | images | pairs matched | graph components | largest component | singleton images | registered | total time |
|---|---|---|---|---|---|---|---|
| F (every 6th) | 55 | 404 | **15** | 22 | 10 | 15 | 200-320 s |
| F3 (every 3rd) | 110 | 639 | **7** | 64 | 3 | 44 | 431 s (matching 275 s) |
| F3r (relaxed) | 110 | 639 | **3** | 95 | 0 | 45 | 351 s |

Findings
- **The loop is cut into disconnected pieces** wherever neighbouring samples share fewer than ~25 matches (corners, foliage, strong sun, scale changes) - heat maps `F/inlier_matrix.png`, `F3/inlier_matrix.png` show strong diagonal blocks separated by gaps. Incremental SfM cannot cross a gap, so the model stays inside one component.
- **Denser sampling helps** (15 -> 44 registered, components 15 -> 7) but costs time: matching grows with images x window (0.43 s per pair), BA grows with model size.
- **Relaxing thresholds did not fix it** (3 components, largest 95 images, but only 45 registered) and caused **no wrong links** (camera error 0.19 %). So connectivity of the pair graph is *necessary but not sufficient*: a new image also needs >= 12-20 matches against points that already exist as 3D tracks. Incremental registration stalls even inside a connected component of 95 images.
- What the registered part is worth: the 44-camera F3 model covers one full side of the building with 0.16 % camera error (`F3/scene.png`, `F3/turntable.mp4`) - accurate where it exists, but incomplete. Not attempted (future work): loop-closure detection by image retrieval, reconstructing components separately and merging them with Sim(3) alignment, re-triangulation passes to grow tracks, GPU matching to afford all pairs.
- No loop closure, so even a full success would drift; with 110 images and BA every 10 registrations this could not be tested.

### 8.6 What went right / wrong - quick ledger
| # | event | result |
|---|---|---|
| 1 | Case A, clean facade | works, 0.03 % |
| 2 | Case C first run | silently wrong (bug), caught by GT diagnosis, fixed |
| 3 | Case B | real failure: occlusion |
| 4 | Case D | works; one image lost to scale change |
| 5 | Stride sweep | robust on this facade up to stride 6; accuracy slips at 8 |
| 6 | Loop F | fail: 15 comps; F3: partial (44/110); F3r: no gain |
| 7 | Runtime | matching dominates: 275 s of 431 s for 639 pairs (-> GPU matching worthwhile) |

## 9. Advanced work: GPU learned-feature matching and COLMAP comparison (2026-10-01)
Hardware / software: RTX 3050 6 GB Laptop GPU, driver 581.86 (CUDA 13.0), PyTorch 2.11.0+cu128, kornia 0.8.3 (replaced the earlier CPU-only torch 2.14). Code: `sfm/gpu_matching.py`, `sfm/colmap_compare.py`, `scripts/{gpu_bench,matcher_bakeoff,diag_run,run_colmap_compare}.py`, `experiments/exp_learned.py`; new cases in `scripts/run_person_hall.py`. Outputs: `results/gpu/`, `results/expL/`, `results/colmap_compare/`, `results/person_hall/*_dl*`.
**Learned models (pretrained, no training done by us):** DISK keypoints+descriptors (Tyszkiewicz et al., NeurIPS 2020; checkpoint `depth-save.pth`, 4.2 MB) and LightGlue matcher (Lindenberger et al., ICCV 2023; `disk_lightglue` checkpoint, 45 MB), loaded through kornia; copies are in `weights/` (see README for sources and licences to verify before redistribution).

### 9.1 GPU port of the classical matcher (a like-for-like speed test)
`match_pair_gpu` = the same ratio-0.8 + mutual-check brute force, with the distance matrix from `torch.cdist` on the GPU.
- **Identical output**: 1,366 matches, all 1,366 identical to the CPU matcher (0 only-CPU, 0 only-GPU) on a south-building pair.
- **10.9x faster per pair**: 0.319 s (CPU, OpenCV BFMatcher) -> 0.029 s (GPU).
- End-to-end the saving is smaller because matching is only part of the pipeline: 12-image control 78 s (CPU) vs 68 s (GPU matcher); the 12-image control reproduces the same reconstruction (10,934 vs 10,938 points; camera error 0.11 % vs 0.12 %, tie-breaking noise).
- Whole loop F3 (639 pairs): CPU matching took 275 s; the learned matcher took 128 s including a transformer (LightGlue 0.15-0.24 s per pair); a GPU brute-force run would need only about 19 s (639 x 0.03 s, extrapolated, not run end to end).

### 9.2 Matcher bake-off on hard pairs, scored with ground-truth geometry (`results/gpu/bakeoff.csv`, `bakeoff_inliers.png`)
Methods: (a) RootSIFT 8000 + ratio/mutual (CPU), (b) DISK 8000 + the *same* ratio/mutual matcher (isolates the benefit of learned **features**), (c) DISK + LightGlue (adds the learned **matcher**). Metric: RANSAC-E inliers (1 px) that are correct under COLMAP's poses (Sampson < 1.5 px).
| pair type | pairs | RootSIFT | DISK + ratio | DISK + LightGlue |
|---|---|---|---|---|
| sun flare (case C) | 1 | 379 | 3,056 | **3,400** |
| foliage (case B), median | 3 | 4 | 28 | **229** |
| scale change (case D), median | 2 | 174 | 809 | **1,124** |
| whole-loop weak links of F3 (< 25 SIFT inliers), median | 13 | 1 | 16 | **54** |
| **weak links bridged (>= 25 correct matches)** | 13 | **0 / 13** | 5 / 13 | **8 / 13** |
| south-building gap 2 (easy) | 1 | 1,270 | 3,912 | 4,203 |
| south-building gap 4 | 1 | 1 | 0 | **1,633** |
| south-building gaps 6, 8, 10, 14 | 4 | ~0 | 0 | ~0 |
Findings: learned **features** give most of the gain on flare / foliage / scale; the learned **matcher** adds a further large step (weak links 16 -> 54; gap-4 pair rescued only by LightGlue). **Limits remain**: viewpoint change beyond roughly gap 4-6 defeats every method (south ladder), 5 of 13 weak links stay at zero for all methods (probably no real overlap), and D 1313-1317 (large scale change) is unmatched by all three. Raw LightGlue precision before RANSAC is only 0.5-0.65 (many wrong matches survive its filter) but RANSAC removes them: surviving inliers were ~100 % correct. Cost: DISK extraction 0.56 s / image vs RootSIFT 0.16 s (CPU).

### 9.3 End-to-end SfM with learned features (same pipeline, only features/matcher changed)
South-building, control sequence (12 images, stride 2) - `results/expL/results.csv`:
| | RootSIFT + BF (CPU) | RootSIFT + GPU matcher | DISK + LightGlue |
|---|---|---|---|
| registered | 12/12 | 12/12 | 12/12 |
| 3D points | 10,934 | 10,938 | **20,084** |
| points seen in >= 3 views | 4,986 | 4,989 | **12,501** |
| verified inliers per pair (mean) | 483 | 483 | **2,245** |
| mean track length | 2.95 | 2.95 | **3.83** |
| mean reprojection error (working px) | **0.138** | 0.138 | 0.379 |
| camera position error | **0.11 %** | 0.12 % | 0.20 % |
| run time | 78 s | 68 s | 116 s |
=> learned matching gives **1.8x the points and 4.6x the inliers, but noisier geometry**: reprojection error 2.7x higher and camera error ~2x higher. DISK keypoints are located on the integer pixel grid (no sub-pixel refinement), SIFT's are sub-pixel.
Viewpoint ladder and few images (south-building, 8 images / N images):
| case | RootSIFT registered, cam err | DISK+LG registered, cam err | DISK+LG with strict pair threshold (>= 150 inliers), registered / correct |
|---|---|---|---|
| stride 3 | 6/8, 0.29 % | **8/8, 0.07 %** | (not needed) |
| stride 4 | 2/8, n/a | 6/8, **85 % (wrong)** | 4 / **4 correct** |
| stride 5 | 3/8, 0.20 % | 4/8, **59 % (wrong)** | 4 / 3 correct |
| stride 6 | 3/8, 0.30 % | 8/8, **74 % (wrong)** | 3 / **3 correct** |
| N=3 images | 3/3, 0.17 % | 3/3, 0.24 % | - |
| N=4 images | 3/4, 0.62 % | **4/4**, 0.35 % | - |
**A second silent failure (same lesson as the orientation bug, 8.2):** with learned matching at stride 4-6 *more* images registered, but the models are wrong (camera error 59-85 %, rotation 20-64 deg). `scripts/diag_run.py` (stride 6): 19 verified pairs, 11 correct (median **2,357** inliers) and 8 wrong (median **68** inliers, 100 % of their matches violate GT geometry); 0 of 8 cameras correct. My SIFT-era thresholds (use pair if >= 25, verified if >= 40 inliers) cannot separate these, because a learned matcher gives true pairs thousands of inliers. Raising both to **150** removes the wrong models (3-4 images registered, essentially all correct). Hypothesis -> test -> confirmed. Even with the fix, learned matching only rescues **moderate** viewpoint change (stride 3-4); it does not beat the fundamental limit (stride 6).

Person Hall (`results/person_hall/cases.csv`):
| case | SIFT | DISK+LG | DISK+LG strict (150) |
|---|---|---|---|
| B foliage | 2/12, 63 pts | 3/12, 2,363 pts, err 0.21 % | 3/12, 2,357 pts - **still FAILURE** |
| D corner turn | 11/12, 5,226 pts, 0.07 % | 10/12, 14,652 pts, 0.29 % | 10/12, 14,649 pts, 0.30 % |
| F loop (every 6th) | 15/55, 0.66 % | 22/55, **44 % (wrong)** | 15/55, **58 % (wrong)** |
| F3 loop (every 3rd) | 44/110, **0.16 %** | 59/110, **53 % (wrong)** | 59/110, **53 % (wrong)** |
Learned matching gives 2-37x more points in B and 2.8x in D, but **did not fix the loop and made the geometry worse**: 15 more cameras registered, but the model is wrong. Diagnosis for F3 strict (`diag_F3_dls*`): 59 registered cameras, only **10 correct**, 49 wrong; **372 of 403 verified pairs have the correct relative pose** (median rotation error 0.62 deg) and the 31 wrong ones still have almost no wrong matches (0.03 %), median 613 inliers, median rotation error 6.1 deg (not gross). Wrong pairs are spread over all index gaps (6-11 %). Camera error grows **smoothly along the chain**: rotation error 2 deg at camera 13, 10 deg at 25, 54 deg at 45, 170 deg at 66; position error from 0.8 % to 100 % of the extent. So: no gross false links; **accumulated drift** from less precise two-view geometry (consistent with the 2.7x higher reprojection error), with no loop closure to correct it. Not isolated by a controlled test (e.g. sub-pixel keypoint refinement, local BA, hybrid SIFT-geometry) - listed as future work; do not claim causation in the report.
Success vs failure here: SIFT F3 registered only 44 cameras but they are accurate (0.16 %); learned F3 registered 59 but 49 are wrong. *Registration count is not a quality measure.*

### 9.4 COLMAP comparison on identical images (`results/colmap_compare/`)
COLMAP's model is the one shipped with the dataset (all images, SIFT, exhaustive/retrieval matching, incremental SfM, BA with intrinsics refinement). Fair comparison: restrict it to the 3D points seen by >= 2 of **our** images and count only those observations; reprojection errors in original-resolution pixels for both. Our intrinsics are COLMAP's (we do not self-calibrate).
| images | ours pts | COLMAP pts (same images) | ours / COLMAP | track length ours / COLMAP | reproj px ours / COLMAP | camera pos err | rot err |
|---|---|---|---|---|---|---|---|
| south 12 (stride 2) | 10,934 | 8,830 | 1.24x | 2.95 / 2.69 | 0.27 / 0.48 | 0.11 % | 0.18 deg |
| south 12, DISK+LG | 20,084 | 8,830 | 2.27x | 3.83 / 2.69 | 0.73 / 0.48 | 0.20 % | 0.28 deg |
| south 16 consecutive | 18,216 | 11,577 | 1.57x | 3.89 / 3.82 | 0.29 / 0.46 | 0.05 % | 0.07 deg |
| hall A | 12,584 | 17,780 | 0.71x | 3.03 / 3.07 | 0.39 / 0.60 | 0.03 % | 0.03 deg |
| hall C | 6,509 | 10,777 | 0.60x | 3.33 / 4.61 | 0.52 / 0.78 | 0.11 % | 0.15 deg |
Reading: (1) **poses agree with COLMAP to 0.03-0.11 % / <= 0.18 deg** - our estimation of E, PnP and BA reproduces a state-of-the-art result when matches are good; this does not prove both are right, only that they agree. (2) South: more points than COLMAP's subset with similar track length; Person Hall: fewer (0.6-0.7x) and shorter tracks in C. (3) Lower reprojection error than COLMAP is *not* evidence of being better: our intrinsics come from COLMAP's refined calibration, COLMAP's error is averaged over all its observations from a model constrained by all images, our filter drops observations above 2 px (working resolution), and point counts depend on thresholds (e.g. 1.5 deg triangulation angle). (4) DISK+LG: 2.3x COLMAP's points but a 51 % higher error. Overlay figures `overlay_south_12.png`, `overlay_hall_A.png`: our red points and COLMAP's blue ones lie on the same structures (C-shaped building footprint in plan view); COLMAP also has points on ground / foliage / far structure where we have none.

What COLMAP does that this project does not (general knowledge of the COLMAP pipeline, not measured here): exhaustive or vocabulary-tree (retrieval) matching, which gives loop-closure candidates; two-view geometry classification (E, F and H models, planar/panoramic detection) before verification; initial-pair selection and **next-best-view** choice by visible-point count and spatial spread; RANSAC triangulation of multi-view tracks and **re-triangulation**; **local BA** after every registration and global BA at growth thresholds; **refinement of intrinsics**; track merging / **multiple sub-models** that can be merged; and a separate dense stage (PatchMatch MVS with photometric+geometric consistency, depth-map fusion, meshing). What we implemented ourselves: two-view E + cheirality, DLT triangulation, PnP registration, tracks, sparse BA with Huber loss, and a rectified-SGBM dense stage with voxel-vote fusion (section 7). Not implemented: loop closure, local BA, intrinsics refinement, sub-model merging - exactly the missing pieces the whole-loop failures (sections 8.5, 9.3) point at.

### 9.5 Ledger of this stage (successes and failures)
| # | event | result |
|---|---|---|
| 1 | CUDA PyTorch + kornia install | worked; torch 2.11.0+cu128 sees the RTX 3050 |
| 2 | GPU brute-force matcher | identical matches, 10.9x faster per pair |
| 3 | DISK + LightGlue weights (49 MB) | downloaded and run; no training needed |
| 4 | Bake-off on hard pairs | big gains on flare / foliage / scale; 8/13 loop weak links bridged; wide baselines still unsolved |
| 5 | Learned SfM, control | 1.8x points but ~2x worse camera error |
| 6 | Learned SfM, stride 4-6 | **silent failure**: more images registered, wrong models (59-85 % error) |
| 7 | Stricter pair threshold (150) | confirmed fix for #6 |
| 8 | Learned SfM, whole loop | **fail**: 59 registered, 49 wrong; drift, not false links; cause not isolated |
| 9 | COLMAP comparison | poses agree to 0.03-0.11 %; point counts 0.6-2.3x depending on case |

## 10. Fix after review: tilted turntables and scene figures (2026-10-01)
**Bug (found by the user while reviewing the videos):** every turntable video and 3D scene figure showed the building lying on its side with the cameras pointing "up". **Cause:** `viz.turntable_video` / `viz.plot_scene` drew points in the SfM world frame, which is the *first camera's* frame (y down, z forward), but matplotlib's 3D axes assume z is up; the azimuth orbit therefore rotated about the cameras' viewing axis instead of the vertical. The interactive HTML viewer had been levelled earlier (section 7) but the matplotlib code had not. **Fix:** `viz.level_matrix(rec)` takes "up" as the mean of the cameras' up axes (-y in camera coordinates; photographers hold cameras roughly upright) and the mean viewing direction as "forward"; all matplotlib drawing code (`plot_scene`, `turntable_video`, the demo-video growth clip) now transforms points and frusta with it, orbits from behind the cameras (azimuth -75 deg at start), and zooms into the scene. Re-rendered: baseline and dense turntables, `scene_views.png`, and (script `scripts/rerender_videos.py`, recorded statistics untouched) the Person Hall scene figures and turntables. **Not re-rendered:** the per-run scene figures `results/exp{A,B,C}/scene_*.png` still use the old tilted view (cosmetic; re-running would overwrite the recorded runtimes). Lesson: look at every figure and video the code produces, not only the numbers.
