# Demo video script - CT3205 Assignment 2

**Target length: 5 minutes** (the brief asks for a "short" demonstration; block 6 is optional, cut it first).
**Worth 10 %** of the marks, and the brief (section 4) says the demonstration **should show**:
1. input images, 2. feature matches, 3. estimated camera positions, 4. the reconstructed 3D point cloud, 5. the ability to rotate / inspect the 3D reconstruction.
Section 6 adds that you are graded on **understanding** (pose estimation, triangulation, how poses are combined, how the cloud is built, how to read the evaluation, why you made your experimental decisions). So the narration must *explain*, not just describe. Every block below lists what is on screen, what to say, and where the material lives.

**Ready-made kit:** `python scripts/build_demo_kit.py --out "C:/Users/jeble/Claude/CT3205_CV_Assignment_2"` builds 1920×1080 slides and clips for every block, plus a silent 5:45 rough cut. The copy of this script in that folder (`DEMO_SCRIPT.md`) points to the kit's files shot by shot. The kit is kept outside the repository because it is about 280 MB; the narration there is the same as here.

**Consistency rule:** every number spoken below is taken from the report (`docs/Report_CT3205_A2_DRAFT.pdf`); the report section is given per block. If you change a number in the report, change it here too. COLMAP is called a *reference*, not ground truth, as in the report.

| # | Time | Block | Brief requirement it covers | Report | Must / optional |
|---|---|---|---|---|---|
| 1 | 0:00-0:25 | Intro and dataset | input images | II-A | must |
| 2 | 0:25-0:55 | Features and matching | feature matches | III-B | must |
| 3 | 0:55-1:45 | Camera pose, triangulation, multi-view | camera positions, understanding | III-C to III-F | must |
| 4 | 1:45-2:20 | Live 3D inspection | 3D cloud, rotate / inspect | V-A | **must** |
| 5 | 2:20-2:45 | Reprojection and accuracy | verification | III-G, V-A | must |
| 6 | 2:45-3:10 | Dense reconstruction | advanced work | V-F | optional |
| 7 | 3:10-3:45 | Experiments A-D | experiments (25 %) | V-B to V-E | must (short) |
| 8 | 3:45-4:25 | Failures and challenging scenarios | scenarios (5 %) | VI | must (short) |
| 9 | 4:25-5:00 | Own photographs | scenarios, advanced | III-K, VI | must (short) |
| 10 | 5:00-5:45 | Learned features, COLMAP, wrap-up | advanced (20 %) | V-G, VIII | must (short) |

(Times follow the narration length at about 130 words per minute: 5:45 with everything in, about 5:20 without block 6. To reach 5:00, also drop the PnP-versus-chaining sentence in block 3 and the AI sentence in block 10.)

**Report figure files:** the file names in `docs/report_figs/` do not all match the report's figure numbers: `fig8_failures.png` is report **Fig. 9** and `fig9_learned_colmap.png` is report **Fig. 8**. The others match.

---

## Block 1 - Intro and dataset (0:00-0:25)
**On screen:** title card, then a quick strip of the 12 input images (`results/demo/demo_auto.mp4` 3-10 s, or `docs/report_figs/fig1_datasets.png`).
**Say:**
"This is multi-view 3D reconstruction. From photos of a static scene, my pipeline recovers where each camera was and a 3D point cloud: features, matching, camera poses, triangulation, bundle adjustment. The main dataset is south-building: twelve photos, scaled to 1600 pixels and undistorted, with the camera matrix taken from the dataset's COLMAP calibration."

## Block 2 - Features and matching (0:25-0:55)
**On screen:** `docs/report_figs/fig2_matching.png` (report Fig. 2), or the three stages one after the other: `results/baseline/matches/matches_P1180189_P1180191_a_raw.jpg`, `_b_ratio_mutual.jpg`, `_c_ransac.jpg` (green = correct, red = wrong under COLMAP's geometry).
**Say:**
"I detect RootSIFT features, 8000 per image, and filter matches in a cascade. Of 8000 raw nearest-neighbour matches, about 4,500 are wrong. Lowe's ratio test and a mutual check leave 3,164, with 81 wrong. RANSAC on the essential matrix keeps 3,077 that agree with one camera motion, and none are wrong. Green and red are judged against COLMAP's epipolar geometry."
**Point to make (understanding):** each filter removes a different failure type; the ratio test removes ambiguous appearance, RANSAC enforces geometry.

## Block 3 - Camera pose, triangulation, multi-view (0:55-1:45)
**On screen:** `results/baseline/two_view_demo.json` shown as a small table (four candidates, points in front of both cameras: **0, 3077, 0, 0**), then the camera-growth clip (`results/demo/demo_auto.mp4` 28-40 s) with cameras appearing one by one.
**Say:**
"The essential matrix gives four possible poses: two rotations times the sign of the translation. Only one puts the triangulated points in front of both cameras, the cheirality test: here 3,077 points for one candidate, none for the others. Triangulation intersects the two viewing rays with a linear SVD solve. The first camera defines the world frame and the first baseline the scale. Every further image is registered by PnP against the existing 3D points, so it lands in the same frame with the same scale; chaining pairwise poses would accumulate error. After each addition, bundle adjustment refines all cameras and points by minimising reprojection error."
**Key words to get right:** cheirality, PnP + RANSAC, one common frame and scale, bundle adjustment (the brief says this is the most important part of the report).

## Block 4 - Live 3D inspection (1:45-2:20)  **do this one live**
**On screen (screen-record):** open `results/viewer/south_building.html` in a browser. Rotate to: front, top-down (the building's corner with the camera arc in front of it), close-up of the facade; switch dense → sparse and toggle the camera frusta.
**What the page shows:** the 16-image dense run (200,000 of the 428,018 dense points, sparse cloud, 16 cameras), **not** the 12-image baseline, and **no COLMAP overlay**. The grey at the top of the building is the slate roof.
**Say:**
"Here is the reconstruction in an interactive viewer: sixteen consecutive images, each camera's position and viewing direction, and the dense point cloud coloured from the photos. I can rotate and zoom. From above you see two walls meeting at the building's corner, and the cameras on an arc in front of them. Switching to the sparse layer shows the 18 thousand structure-from-motion points the dense stage was built on."
**Alternative:** `python scripts/view_open3d.py results/baseline --gt` shows the 12-image baseline (10,934 points, 12 cameras) with COLMAP's points in grey after Sim(3) alignment. If you record that instead, say "10,934 points and twelve cameras" and "the grey points are COLMAP's reconstruction, aligned to mine".
**Tip:** move the mouse slowly; stop for 2 seconds on each view.

## Block 5 - Reprojection and accuracy (2:20-2:45)
**On screen:** `docs/report_figs/fig4_reprojection.png` (report Fig. 4: green = detected, red = reprojected, offset magnified x15, plus the histogram), or `results/baseline/reproj_P1180193.jpg` and `results/baseline/reproj_hist.png`.
**Say:**
"I project every 3D point back into each image: the mean error is 0.14 pixels. A large error would mean a wrong pose, a bad point or a false match. Two-view points fit almost by construction, so I also report three-view points: 0.16 pixels. Camera positions agree with COLMAP to 0.11 percent of the trajectory extent."
**Numbers (report Table I, `results/baseline/stats.json`):** 12/12 registered, mean 0.138 px, median 0.103 px, >=3-view mean 0.158 px, camera position 0.11 %, rotation 0.18 deg mean / 0.37 deg max, 99.4 % of points within 2 % of a COLMAP point.

## Block 6 - Dense reconstruction (2:45-3:10, optional)
**On screen:** `docs/report_figs/fig7_dense.png` or `results/dense/sparse_vs_dense.png`, then `results/dense/dense_turntable.mp4`.
**Say:**
"With accurate poses I can go dense: rectify neighbouring pairs, match every pixel with semi-global stereo, and fuse the depth maps, keeping a point only where two pairs agree. Sixteen images go from 18 thousand to 428 thousand points, 23 times more. Individual dense points are about a third noisier, but they cover more of COLMAP's surface."

## Block 7 - Experiments A-D (3:10-3:45)
**On screen:** `docs/report_figs/fig5_expA_expC.png` (A and C) and `docs/report_figs/fig6_expD.png` (D), plus `results/expD/degradation_examples.jpg` for 3 s. For B, show report Table III or `results/expB/expB_metrics.png`.
**Say (one sentence each):**
- A: "More images give more points, from 1.4 to 23 thousand, but accuracy is best at 8 images, not 20."
- B: "RootSIFT was the best of five detectors; ORB found four times fewer inliers and registered 8 of 12."
- C: "Matching collapses near 35 degrees of viewpoint change, and the smallest baseline gives the weakest depth."
- D: "Blur is the most destructive: sigma 4 leaves 7 percent of the points; darkening barely matters."
**Why this matters for the grader:** it shows you chose variables, controlled everything else, and can interpret the result.

## Block 8 - Failures and challenging scenarios (3:45-4:25)
**On screen:** `results/person_hall/exif_bug_before_after.png` (5 s), `results/person_hall/B/matches/matches_IMG_1045_IMG_1048_c_ransac.jpg` (4 s), `results/person_hall/F/inlier_matrix.png` (5 s), `results/person_hall/F3/turntable.mp4` (6 s). Report Table VI and Fig. 9 (`docs/report_figs/fig8_failures.png`) hold the same evidence.
**Say:**
"On Person Hall, a tree leaves 74 correct but too few matches, and only 2 of 12 images register. More instructive: one test registered all twelve images with low reprojection error, yet was 51 percent wrong. The cause was my loader: OpenCV applied an EXIF rotation that COLMAP ignores. Only the comparison with COLMAP caught it. Around the whole building the image-pair graph falls into disconnected clusters; relaxed thresholds reconnect most of it, but only 45 of 110 images register, so only one side is recovered."

## Block 9 - Own photographs (4:25-5:00)
**On screen:** `results/own/yellow_all/turntable_object.mp4`, `results/own/snacks/turntable_object.mp4`, then the negative case `results/own/bottle_classic/turntable_object.mp4` (each about 5 s); stills `docs/report_figs/fig10_own_positive.png` and `fig11_own_negative.png`; optionally `results/own/calib_photo_review.jpg` (1-2 s) when the chessboard is mentioned. Interactive pages: `results/viewer/own_yellow_all_object.html`, `own_fire_object_v2.html`, `own_snacks_object.html`, `own_bottle_classic_object.html`.
**Say:**
"I photographed four objects with my phone, 330 photos, including a transparent bottle on purpose. The phone is uncalibrated, so I estimated its focal length from fundamental matrices; that method recovers COLMAP's value to 0.5 percent, and a chessboard agrees to about 1 percent. All 330 photos register on near-circular orbits, and the sign's text is legible in 3D. The bottle shows the limit: only the opaque label and cap reconstruct, and learned features did not help."
**Do not claim a metric accuracy for these sets:** the chessboard checks the focal length, not the reconstructions.

## Block 10 - Learned features, COLMAP, wrap-up (5:00-5:45)
**On screen:** `results/gpu/bakeoff_inliers.png` (6 s), `results/colmap_compare/overlay_south_12.png` (6 s; both together are report Fig. 8, `docs/report_figs/fig9_learned_colmap.png`), closing title.
**Say:**
"I added learned features, DISK with LightGlue, on the GPU: 3,400 correct matches against 379 under sun flare, and 8 of 13 weak loop links bridged where SIFT bridged none. But their geometry is noisier, and on the loop 49 of 59 registered cameras were wrong. Against COLMAP on identical images my poses agree to 0.03 to 0.11 percent. In short: the pipeline matches COLMAP when matches are good, and the failures taught the most: reprojection error and registration counts can hide a wrong reconstruction. AI tools helped with code and writing; the declaration is in the report."

---

## Recording checklist
- **Tools:** Windows Game Bar (Win+G) or OBS Studio; 1080p, 30 fps, microphone check first. Record narration separately in Audacity if easier, then sync.
- **Browser/windows:** hide bookmarks and notifications; zoom the browser to 125 % so the HTML viewer is readable.
- **Auto-generated clip (tested, 2026-10-01):** `python scripts/make_demo_video.py --start 40 --count 12 --stride 2 --out results/demo` writes `results/demo/demo_auto.mp4` (58 s, 1280x720, 24 fps, **no audio**, about 1 minute to run). Contents in order: title (0-3 s); the 12 input images (3-10 s); match stages raw -> ratio+mutual -> RANSAC for three image pairs with counts in the title bar (10-27 s); cameras added one by one with a live point count (28-40 s); a 360-degree turntable (40-52 s); reprojection overlays with error histograms (50-58 s). Use it for blocks 1, 2, 3 (growth part) and 5. **Known limits:** it is a sparse-only clip (no dense, no experiments, no failures, no own photos); the reprojection frames have small text (zoom in your editor). It is raw material, not the finished demo: add your voice and the live inspection (block 4).
- **Order of recording:** (1) screen-record block 4 live (and optionally one own-object HTML page for block 9); (2) assemble the still figures and turntables with a free editor (Clipchamp, DaVinci Resolve, or Windows Photos video editor); (3) record voice-over last, reading the script above in your own words.
- **Pace:** about 130 words per minute. The narration is about 710 words (650 without block 6), about 5.5 minutes of speech plus pauses for the live inspection; say it in your own words rather than reading it.

## Say it in your own words (the brief tests understanding)
Be ready to answer without notes:
- Why four pose candidates, and how is one chosen?
- What does a large reprojection error mean?
- Why do two-view points have artificially small errors?
- How do images end up in one coordinate frame, and why PnP instead of chaining relative poses?
- What does bundle adjustment minimise?
- Why did the loop fail even after the thresholds were relaxed?
- How did you get the phone's focal length, and how do you know it is right?

The answers are in the report (Sections III, VI and VII) and in `docs/PROGRESS.md`.

## Things NOT to claim (to stay honest and safe under questioning)
- Do not call COLMAP "ground truth": it is a reconstruction too, and our intrinsics for the COLMAP datasets are its calibration. Say "agrees with COLMAP".
- Do not say learned matching "fixed" the loop: it did not (59 registered, 49 wrong). Say the errors accumulate along the chain without loop closure and that the cause was **not isolated**.
- Do not say our reprojection error is "better than COLMAP": our camera calibration comes from COLMAP and the metrics are not identical.
- Do not present the first run of hall case C or the stride 4-6 learned results as successes: they are examples of silent failure.
- Do not claim a metric accuracy for the own captures: the chessboard validated the focal length (about 1 %), not the 3D models.
