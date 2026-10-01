# Report skeleton - CT3205 Assignment 2 (max 15 pages, excluding references and appendices if permitted)

**How to use this file.** It follows the section list in the brief (section 7: Introduction, Dataset, Methodology, Experimental Design, Results, Challenging Scenarios, Discussion, Conclusion, AI Usage Declaration). Each section gives: a page budget, the points to make, the exact figures / tables to include (with file paths), and the numbers (all copied from `docs/PROGRESS.md`, which cites the result files). Write the prose yourself in your own words, since you are graded on understanding. Items in **[YOU]** need information only you have. Items marked **(!)** are honest caveats you should keep in the report.
Deadline: 4 October 2026 (start of Week 8). Assessment weights: baseline 40 %, experiments 25 %, challenging scenarios 5 %, demo 10 %, advanced 20 %. The brief says Methodology is **the most important part**, specifically "multi-view reconstruction".

## Page budget (target 14.5 of 15)
| Section | Pages | Weight it serves |
|---|---|---|
| 1 Introduction | 0.75 | all |
| 2 Dataset | 1.0 | baseline |
| 3 Methodology | **4.5** | baseline 40 % + advanced |
| 4 Experimental design | 1.0 | experiments 25 % |
| 5 Results | 3.5 | experiments, baseline, advanced |
| 6 Challenging scenarios | 1.5 | scenarios 5 % |
| 7 Discussion | 1.25 | experiments, scenarios |
| 8 Conclusion | 0.5 | all |
| 9 AI Usage Declaration | 0.5 | required |
| Appendix (not counted if permitted) | - | reproduction commands, versions, extra tables |
Tip: figures carry most of the weight. Make every figure a *multi-panel* figure (matches before / after on one page) and cite it in the text.

---

## 1. Introduction (0.75 page)
- **Problem:** multi-view 3D reconstruction from a sequence of photos of a static scene: recover camera poses and a 3D point cloud from images alone. Extension of the two-view stereo lab to N views.
- **Objectives (the brief's six outputs):** correspondences, relative camera poses, 3D points, a sparse cloud, camera positions and orientations, reprojection of points into the images.
- **What was built:** own incremental SfM pipeline (features, matching + verification, essential matrix, cheirality, DLT triangulation, PnP registration, sparse bundle adjustment), evaluated against COLMAP ground truth; plus advanced work (dense stereo; GPU learned features; COLMAP comparison).
- **Key findings in 4-5 bullets (use these numbers):** baseline 12/12 images, 10,934 points, camera position error 0.11 % and rotation 0.18 deg vs COLMAP; RootSIFT beat SIFT/ORB/AKAZE/BRISK; matching collapses beyond ~35 deg viewpoint change; a "successful" reconstruction that was 51 % wrong (EXIF bug) shows why independent validation is needed; learned matching finds 2-10x more correct matches on hard pairs but gave noisier geometry and did not fix the whole-building loop.

## 2. Dataset (1.0 page)
**[YOU]** If you add your own photos, describe them *here*: camera/device, image resolution, number of images, approximate camera movement, moved around the object or object rotated (the brief requires all five).
- **south-building (main):** COLMAP dataset, 128 photos, 3072x2304, SIMPLE_RADIAL, f = 2559.7 px, k1 = -0.0205. Sub-sequences used: images P1180181-P1180203 (12 images, stride 2) as the baseline and control; 16 consecutive images (P1180181-196) for dense. Images downscaled to longest side 1600 px; lens distortion removed; intrinsics taken from the dataset's COLMAP calibration (we do not self-calibrate (!)).
- **person-hall (challenging):** 330 photos, 5616x3744, OPENCV model (k1 = -0.109), a closed walk around a brick building; heavy foliage, repetitive arched windows, sun flare; **48 photos carry an EXIF portrait tag**.
- **Overlap:** state the neighbouring-image overlap qualitatively (consecutive south-building frames have ~12 deg parallax; matching survives up to ~35 deg).
- **Figures:** Fig 1 = strip of the 12 input images (crop from `results/demo/demo_auto.mp4` or hand-assemble from `Datasets/south-building/images/`); Fig 2 = `results/person_hall/contact_sheet.jpg`.
- **(!)** Ground truth = the COLMAP model shipped with the datasets; it is itself a reconstruction, so "error" means "disagreement with COLMAP".

## 3. Methodology (4.5 pages) - the most important section
One subsection per brief item; keep the order below. Use equations sparingly but include those marked with an equation symbol.

### 3.1 Feature detection and description (0.4 p)
RootSIFT (SIFT with L1-normalised, square-rooted descriptors), 8000 keypoints per image, contrast threshold 0.02. **Report the number of detected features** (8000 per image, mean) and the matching method. Detectors compared in Exp. B: SIFT, RootSIFT, ORB, AKAZE, BRISK (OpenCV 5 moved AKAZE/BRISK to `cv2.xfeatures2d`).

### 3.2 Matching and filtering (0.7 p)
Cascade: raw nearest neighbour -> Lowe ratio test 0.8 -> mutual (cross-check) -> RANSAC on the essential matrix with 1 px threshold. **Figure 3 (required "before / after" and correct / incorrect matches):** the three panels `results/baseline/matches/matches_P1180189_P1180191_{a_raw,b_ratio_mutual,c_ransac}.jpg`; red = violates COLMAP's epipolar geometry, green = correct (read counts from each title bar). State parameters: ratio 0.8, RANSAC prob 0.999, thr 1 px.

### 3.3 Essential matrix and relative pose (0.7 p)
- Equation: E = [t]x R; estimated by RANSAC from matched points and K (5-point solver in OpenCV).
- **Ambiguity:** SVD of E gives four (R, t) candidates: {R1, R2} x {+t, -t}. **Cheirality:** only one puts the triangulated points in front of both cameras.
- **Table 1 (evidence):** `results/baseline/two_view_demo.json`: pair P1180189-P1180191, points in front of both cameras for the four candidates: **0, 3077, 0, 0**; chosen rotation 14.0 deg; agrees with OpenCV `recoverPose` (rotation error 0.0 deg); our DLT triangulation agrees with OpenCV's to 8e-12.

### 3.4 Multi-view camera pose estimation (1.4 p) **-> the part the brief weighs most**
Explain, in this order: (1) the common frame: first camera P1 = K[I|0], world = frame of the initial pair's first camera, scale = baseline of that pair (monocular scale ambiguity). (2) Choosing the initial pair: candidates ranked by verified inliers, then by median parallax (>= 3 deg required) and number of valid points. (3) Registering each new image: 2D-3D correspondences through its matches to already-registered images, **PnP + RANSAC** (EPnP, 3 px) then LM refinement, so each camera pose P_i = K[R_i | t_i] is expressed directly in the common frame. (4) **Why PnP instead of chaining relative poses:** chained relative transforms R_i = R_(i,i-1) R_(i-1) accumulate error and have unknown per-pair scale; PnP against the shared 3D points keeps scale consistent. (5) Tracks (a 3D point and the keypoint observing it in each image) and how new points extend existing tracks. (6) Bundle adjustment after each registration (3.6). **Figure 4:** growth clip frames or `results/baseline/scene_views.png` (cameras + cloud). Mention registration order from the log.

### 3.5 Triangulation (0.4 p)
Linear DLT in normalised coordinates: rows x*P3 - P1 and y*P3 - P2 from both views, smallest singular vector; **explain how the projection matrices are used.** Validity checks: positive depth in both cameras (physically meaningful depth), reprojection error < 2 px, parallax > 1.5 deg.

### 3.6 Bundle adjustment (0.3 p)
Minimise the sum of Huber-robust squared reprojection errors over all camera poses and 3D points (first camera fixed to remove the gauge freedom), sparse Jacobian (SciPy `least_squares`, TRF). Typical effect: RMSE 0.22 -> 0.14 px per step (see log).

### 3.7 Point cloud, visualisation, reprojection (0.3 p)
Colours sampled from images; PLY export; matplotlib / Open3D / the self-contained HTML viewer (`results/viewer/south_building.html`). **Reprojection:** x_i ~ P_i X; compare with detected feature. **State what a large error means:** wrong pose, wrong 3D point or a false match. **(!)** Points seen in only two views fit by construction, so report errors for >= 3-view tracks as well.

### 3.8 Evaluation protocol (0.3 p)
Sim(3) alignment (Umeyama, RANSAC, optical-axis fix for near-collinear cameras) to COLMAP; camera position error relative to the trajectory extent, rotation error in degrees, share of points within 2 % of a COLMAP point, completeness; match correctness from GT epipolar geometry. **(!)** with few collinear cameras the alignment itself is ill-posed (N=4 case).

### 3.9 Advanced: dense stereo (0.4 p)
Rectify neighbouring image pairs with the estimated poses, SGBM (block 5), back-project, fuse by voxel voting (>= 2 pairs agree). **Figure:** `results/dense/stereo_pair.jpg` (rows aligned) and sparse-vs-dense.

### 3.10 Advanced: learned features on the GPU (0.4 p)
DISK keypoints/descriptors and LightGlue matcher (pretrained, via kornia; weights in `weights/`), RTX 3050, PyTorch 2.11 + CUDA 12.8; GPU port of the classical matcher (identical output, 10.9x faster per pair); stricter pair thresholds (>= 150 inliers) because learned matchers give true pairs thousands of inliers.

## 4. Experimental design (1.0 page)
One compact table (**Table 2**): experiment, variable changed, controls (everything else = the baseline configuration), purpose, hypothesis. Take the hypotheses from `docs/PROGRESS.md` section 5, written **before** the results, then state in Results whether each held.
| Exp | Variable | Control | Purpose |
|---|---|---|---|
| A | number of images N = 3,4,6,8,12,16,20 (nested subsets, stride 2) | RootSIFT, 1600 px, same pipeline | effect of image count |
| B | detector: SIFT, RootSIFT, ORB, AKAZE, BRISK | same 12 images | effect of features |
| C | frame stride 1-6 with 8 images (+ 2-view baseline study) | same detector | viewpoint change |
| D | blur, noise, resolution, contrast, gamma | same 12 images | image quality |
| Challenging | 11 Person Hall cases (A-F3r) | default pipeline | difficult scenes |
| Advanced | detector/matcher (classical vs GPU vs learned) | same sequences | learned features |
Also state the metrics: points, track length, reprojection error (>= 2- and 3-view), camera pos / rot error, completeness, runtime.

## 5. Results (3.5 pages)
### 5.1 Baseline (0.8 p) - Table 3 + Fig 5
12 images P1180181-203: **12/12 registered, 10,934 points** (4,986 seen in >= 3 views), mean track 2.95; reprojection mean 0.138 px, median 0.103, >= 3-view 0.158; camera position error 0.11 %, rotation 0.18 deg (max 0.37); 99.4 % of points within 2 % of a COLMAP point; runtime 66 s. Figures: `results/baseline/scene_views.png` (3D cloud + camera position and orientation, **required**), `results/baseline/reproj_P1180193.jpg` + `reproj_hist.png` (original vs reprojected points and error histogram, **required**).
### 5.2 Experiments A-D (2.1 p) - one table or figure each; 2-3 sentences of interpretation
- **A** (`results/expA/results.md`, `expA_metrics.png`): points 1.4k -> 23k, completeness 4 -> 63 %, runtime ~N^2 (212 s at N=20); **accuracy not monotonic: best at N=8 (0.08 %), worse at 16/20 (0.4 / 0.3 %)**; reprojection error rises 0.11 -> 0.18 px because of longer tracks; **N=4 is a fragile reconstruction (3/4 registered, accuracy metric unreliable)** (!).
- **B** (`expB_metrics.png`): RootSIFT 12/12, 10,934 pts, 483 inliers/pair; SIFT 11/12, 9,414; ORB 8/12, 3,706, reproj 0.35 px, ~10x worse camera error; AKAZE 8/12; BRISK 9/12. Binary detectors untuned (!).
- **C** (`expC_metrics.png`): registered 8,8,6,2,3,3 of 8 for strides 1-6; two-view study: 2,520 / 1,270 / 377 inliers at parallax 12.6 / 24 / 34 deg, noise beyond; **south-building never reaches the true small-baseline regime** (!).
- **D** (`expD_metrics.png`, `degradation_examples.jpg`): graceful then collapse: blur sigma 4 leaves 7 % of the points; noise 50 -> 2/12 registered; resolution 200 px -> 6/12; contrast x0.1 fails; gamma harmless.
### 5.3 Dense and learned features (0.6 p)
- Dense: 18,216 -> 428,018 points (23.5x); COLMAP points within 2 % of scene radius covered 58.5 -> 66.4 %; dense points ~35 % noisier than sparse; ablation: fusion support is the main knob (46 % vs 80 % of points on the facade plane for support 1 vs 3). `results/dense/sparse_vs_dense.png`, `ablation.csv`.
- Learned features (`results/expL/results.md`, `results/gpu/bakeoff_inliers.png`): control run 20,084 vs 10,934 points but camera error 0.20 % vs 0.11 %; hard pairs: sun flare 379 -> 3,400 correct matches, foliage 4 -> 229, weak loop links bridged 0/13 -> 8/13.
- COLMAP comparison (`results/colmap_compare/comparison.md`, `overlay_south_12.png`): poses agree to 0.03-0.11 %; points 0.6-2.3x COLMAP's. **(!)** lower reprojection error than COLMAP is not evidence of being better (our K comes from COLMAP; metrics differ).
Do not paste raw tables of every case here; put them in the appendix.

## 6. Challenging scenarios (1.5 pages) - **include all failures; the brief grades failure analysis**
Present **Table 4** (Person Hall cases, from `results/person_hall/cases.csv`): A clean facade SUCCESS (12/12, 0.03 %); **B foliage FAILURE (2/12, 63 points)**; C repetitive arches + flare SUCCESS after fix; D corner turn SUCCESS (11/12); stride sweep 2-8 all succeed, accuracy slips at 8 (0.75 %); **F whole loop FAILURE (15/55)**, **F3 FAILURE (44/110, accurate but partial)**.
- **Failure 1: foliage.** Figure `results/person_hall/B/matches/..._c_ransac.jpg`: a tree hides the wall; the 74 surviving matches are all correct but too few.
- **Failure 2 (best story): "success" that was wrong.** `results/person_hall/exif_bug_before_after.png`: 12/12 registered, 0.6 px reprojection error, **51 % camera error / 139 deg rotation error**. Cause: EXIF orientation tags on 48 photos (OpenCV rotates, COLMAP does not). Lesson: validate against independent evidence.
- **Failure 3: the loop.** `results/person_hall/F/inlier_matrix.png`: 15 disconnected clusters (every 6th image), 7 with every 3rd; incremental SfM cannot cross the gaps; relaxing thresholds did not help (connectivity necessary, not sufficient). With learned matching, 59 cameras registered but **only 10 correct** (drift; cause not isolated (!)).
- **Case D:** one image lost (IMG_1313), likely a large scale change.
- **[YOU]** add your own small object / building here if you capture them, plus the "tiny camera step" case showing failed depth.

## 7. Discussion (1.25 pages)
Organise by the brief's list: factors affecting quality, viewpoint and feature effects, reconstruction errors, missing regions, limitations, possible improvements.
- **Viewpoint/features:** parallax is the real variable (not frame stride); too small -> poor depth (stride 1 accuracy 0.917 vs 0.992), too large -> no matches (~35 deg). Learned features move the limit slightly, not fundamentally.
- **Reconstruction errors:** reprojection error is necessary but not sufficient (two-view points, wrong-but-consistent models); registration count is not quality (SIFT F3: 44 cameras accurate; learned F3: 59 cameras, 49 wrong); loop drift without loop closure.
- **Missing regions:** foliage, sky, glass; dense stereo drops foliage and thin structures.
- **Limitations:** one main sequence; RANSAC randomness not averaged; COLMAP is a reference, not truth; intrinsics not estimated; no loop closure, no local BA.
- **Improvements:** loop-closure by image retrieval; local BA and sub-model merging (what COLMAP does); sub-pixel refinement of learned keypoints; semantic masking of vegetation; denser sampling; GPU matching for all pairs.

## 8. Conclusion (0.5 page)
Three or four sentences: pipeline works and agrees with COLMAP (0.1 %); the most important lessons are about *evaluation* (reprojection error can hide wrong models; EXIF bug; registration count) and *limits* (viewpoint ~35 deg, foliage, loops); learned features help on hard pairs but are not a free upgrade.

## 9. AI Usage Declaration (0.5 page) - mandatory; four items required by the brief
**[YOU] must review and own every statement.** Starting draft (edit freely):
1. **Which tools:** Claude (Claude Code with Claude Sonnet 5.5) for coding, analysis and drafting. List any other tool you used.
2. **What for:** writing and debugging the SfM pipeline and experiment scripts, generating figures and videos, diagnosing failures, drafting documentation.
3. **Which parts were AI-assisted:** most code in `sfm/`, `scripts/`, `experiments/`; the progress log; draft of this report. Say what you wrote yourself.
4. **How verified or modified:** every result was checked against COLMAP ground truth; cross-checks against OpenCV (pose recovery, triangulation); bugs found by inspecting failures (e.g. EXIF orientation); you ran the code and reviewed the figures. **[YOU]** say what you read, changed and can explain.

## Appendix (not counted if permitted)
A. Reproduction commands (README); B. library versions (`scripts/check_env.py`, `requirements.txt`); C. full tables: `results/expA..D/results.md`, `results/person_hall/cases.csv`, `results/gpu/bakeoff.csv`; D. file layout.

---

## Requirement coverage checklist (brief -> where covered -> evidence)
| Brief requirement | Covered in | Evidence file |
|---|---|---|
| >= 6 images, same static scene | 2, 5.1 | 12-image baseline |
| Report camera/device, resolution, number, movement, around/rotated | 2 | **[YOU] for own photos**; datasets described |
| 1.1 detector, features count, matching method, parameters; compare detectors | 3.1, Exp B | `results/expB/` |
| 1.2 matching + filtering; show matches before/after, correct/incorrect | 3.2, Fig 3 | `results/baseline/matches/` |
| 1.3 E, decomposition, ambiguity, cheirality | 3.3 | `two_view_demo.json` |
| 1.4 multi-view poses, common frame, visualise cameras | 3.4, Fig 4 | `scene_views.png` |
| 1.5 triangulation, projection matrices, depth check | 3.5 | `sfm/geometry.py` |
| 1.6 multi-view point cloud + camera positions + viewing directions | 5.1 | `scene_views.png`, HTML viewer |
| 1.7 reprojection, original vs reprojected vs error, meaning of large error | 3.7, 5.1 | `reproj_*.jpg`, `reproj_hist.png` |
| >= 3 experiments (A-D) | 4, 5.2 | `results/exp{A,B,C,D}/` |
| >= 3 challenging scenarios incl. small object to building | 6 | Person Hall cases; **[YOU] small object (own photos)** |
| Demonstration video | - | `docs/DEMO_SCRIPT.md`; **[YOU] record** |
| Advanced (dense, learned features, COLMAP comparison with analysis) | 3.9, 3.10, 5.3 | `results/dense/`, `results/gpu/`, `results/colmap_compare/` |
| Private GitHub repo, README, weights, AI declaration | - | repo `Jeb-Lee/CT3205_CV_02` (private), `README.md`, `weights/`, section 9 |

## Gaps to close before submission (honest status)
1. **[YOU] Small object / own photos:** the brief says "from a small object to a building"; we only have buildings. Capture one small textured object (guide: `docs/GUIDE_YOUR_TODO.md`) and run it through the pipeline; I can run it once the photos are in `Datasets/<name>/images/`.
2. **[YOU] Demo video** (script ready), **AI declaration** (draft above), **report text**.
3. The per-run scene plots in `results/expA..C/scene_*.png` have the old tilted view; use `results/baseline/scene_views.png` and the re-rendered figures instead.
4. The brief lists "trained model weights": we use pretrained DISK/LightGlue (copies in `weights/`); say clearly that no training was done.
