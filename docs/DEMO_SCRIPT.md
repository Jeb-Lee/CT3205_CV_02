# Demo video script (proposal) - CT3205 Assignment 2

**Target length: 5 minutes** (the brief asks for a "short" demonstration; trim the *optional* blocks if you need 4).
**Worth 10 %** of the marks, and the brief (section 4) says the demonstration **should show**:
1. input images, 2. feature matches, 3. estimated camera positions, 4. the reconstructed 3D point cloud, 5. the ability to rotate / inspect the 3D reconstruction.
Section 6 adds that you are graded on **understanding** (pose estimation, triangulation, how poses are combined, how the cloud is built, how to read the evaluation, why you made your experimental decisions). So the narration must *explain*, not just describe. Every block below lists what is on screen, what to say, and where the material lives.

| # | Time | Block | Brief requirement it covers | Must / optional |
|---|---|---|---|---|
| 1 | 0:00-0:25 | Intro and dataset | input images | must |
| 2 | 0:25-1:05 | Features and matching | feature matches | must |
| 3 | 1:05-1:50 | Camera pose, triangulation, multi-view | camera positions, understanding | must |
| 4 | 1:50-2:30 | Live 3D inspection | 3D cloud, rotate / inspect | **must** |
| 5 | 2:30-3:00 | Reprojection and accuracy | verification | must |
| 6 | 3:00-3:30 | Dense reconstruction | advanced work | optional |
| 7 | 3:30-4:10 | Experiments A-D | experiments (25 %) | must (short) |
| 8 | 4:10-4:45 | Failures and challenging scenarios | scenarios (5 %) | must (short) |
| 9 | 4:45-5:15 | Advanced work and wrap-up | advanced (20 %) | must (short) |

(Block times add to 5:15 with everything in; cut block 6 or shorten 7-9 to land at 5:00.)

---

## Block 1 - Intro and dataset (0:00-0:25)
**On screen:** title card, then a quick strip of the 12 input images (`results/demo/demo_auto.mp4` first 15 s, or flip through `Datasets/south-building/images/P1180181.JPG ... P1180203.JPG`).
**Say (about 60 words):**
"This is multi-view 3D reconstruction. From a sequence of photos of a static scene, my pipeline recovers where each camera was and a 3D point cloud, step by step: features, matching, camera poses, triangulation, bundle adjustment. I use the south-building dataset: twelve photos, 3072 by 2304, downscaled to 1600 pixels, with the camera matrix taken from the dataset's COLMAP calibration."

## Block 2 - Features and matching (0:25-1:05)
**On screen:** the three match figures of one pair, one after the other: `results/baseline/matches/matches_P1180189_P1180191_a_raw.jpg`, `_b_ratio_mutual.jpg`, `_c_ransac.jpg` (counts are in the title bar of each image; green = correct, red = wrong under COLMAP's geometry).
**Say:**
"I detect RootSIFT features, 8000 per image. Matching goes through a cascade. Raw nearest-neighbour matches contain many mistakes. Lowe's ratio test and a mutual check remove ambiguous ones. Finally RANSAC on the essential matrix keeps only matches that agree with one camera motion. Green lines are correct against COLMAP's ground-truth geometry, red are wrong; after RANSAC none are wrong here."
**Point to make (understanding):** each filter removes a different failure type; RANSAC enforces geometry, not appearance.

## Block 3 - Camera pose, triangulation, multi-view (1:05-1:50)
**On screen:** `results/baseline/two_view_demo.json` shown as a small table (four candidates, points in front of both cameras: **0, 3077, 0, 0**), then the camera-growth clip (`results/demo/demo_auto.mp4` "incremental" segment) with cameras appearing one by one.
**Say:**
"From the essential matrix E equals t-cross R, an SVD gives four possible camera poses: two rotations times the sign of the translation. Only one is physically valid, the one that puts the triangulated points in front of both cameras. This is the cheirality test: here 3077 points are in front for exactly one candidate and none for the other three. Triangulation then finds each 3D point as the intersection of the two viewing rays, by solving a linear system with the SVD. The first camera defines the world frame, P1 equals K times identity and zero. Every further image is added by PnP: it matches its features to existing 3D points, estimates its pose with RANSAC, and then triangulates new points. After each addition, bundle adjustment refines all cameras and points together by minimising reprojection error. That is how all poses end up in one common coordinate system."
**Key words to get right:** cheirality, PnP + RANSAC, bundle adjustment, common coordinate system (the brief says this is the most important part of the report).

## Block 4 - Live 3D inspection (1:50-2:30)  **do this one live**
**On screen (screen-record):** open `results/viewer/south_building.html` in a browser, or run `python scripts/view_open3d.py results/baseline --gt`. Rotate to: front, top-down (shows the L-shaped footprint with the camera arc around it), close-up of the facade; toggle the camera frusta and the sparse / dense layers.
**Say:**
"Here is the result: 10,934 3D points, coloured from the photos, and the twelve camera positions and viewing directions. I can rotate and zoom. From above you see the building's L-shaped footprint and the cameras on an arc around it. Grey points, when I switch the overlay on, are COLMAP's reconstruction."
**Tip:** move the mouse slowly; stop for 2 seconds on each view.

## Block 5 - Reprojection and accuracy (2:30-3:00)
**On screen:** `results/baseline/reproj_P1180193.jpg` (green = detected, red = reprojected, yellow = error lines) and `results/baseline/reproj_hist.png`.
**Say:**
"To check the reconstruction I project every 3D point back into each image and compare it with the detected feature. The mean error is 0.14 pixels, the median 0.10. A large reprojection error would mean a wrong pose, a wrong 3D point, or a false match. One honest caveat: points seen in only two images fit almost perfectly by construction, so for points seen in three or more views I also report 0.16 pixels. Against COLMAP's ground truth, our camera positions differ by 0.11 percent of the trajectory length and rotations by 0.18 degrees."
**Numbers (from `results/baseline/stats.json`):** 12/12 registered, mean 0.138 px, median 0.103 px, >=3-view mean 0.158 px, camera position error 0.11 %, rotation 0.18 deg, 99.4 % of points within 2 % of a COLMAP point.

## Block 6 - Dense reconstruction (3:00-3:30, optional)
**On screen:** `results/dense/sparse_vs_dense.png`, then `results/dense/dense_turntable.mp4`.
**Say:**
"Because the camera poses are accurate, I can go dense: rectify neighbouring image pairs, match every pixel with semi-global stereo, and fuse the depth maps, keeping only points confirmed by at least two pairs. That turns 18 thousand points into 428 thousand, 23 times more, and the facade, windows and bushes become visible. The dense points are about a third noisier than the sparse ones, which I measured."

## Block 7 - Experiments A-D (3:30-4:10)
**On screen:** the four summary figures one after the other: `results/expA/expA_metrics.png`, `results/expB/expB_metrics.png`, `results/expC/expC_metrics.png`, `results/expD/expD_metrics.png` (plus `results/expD/degradation_examples.jpg`, 3 s).
**Say (one sentence each):**
- A: "More images give more points, from 1.4 thousand to 23 thousand, but not always better accuracy: camera error is best at 8 images and worse at 16 and 20. And with 4 images the reconstruction was fragile."
- B: "Of five detectors, RootSIFT was best; ORB found four times fewer inliers and registered only 8 of 12 images."
- C: "Matching collapses beyond about 35 degrees of viewpoint change."
- D: "Blur is the most destructive: sigma 4 leaves 7 percent of the points. Darkening barely matters."
**Why this matters for the grader:** it shows you chose variables, controlled everything else, and can interpret the result.

## Block 8 - Failures and challenging scenarios (4:10-4:45)
**On screen:** `results/person_hall/exif_bug_before_after.png` (5 s), `results/person_hall/B/matches/matches_IMG_1045_IMG_1048_c_ransac.jpg` (4 s), `results/person_hall/F/inlier_matrix.png` (5 s), `results/person_hall/F3/turntable.mp4` (8 s).
**Say:**
"On the Person Hall dataset I tested difficulties. A tree in front of the wall reduced matches to 74 and the reconstruction failed. A more instructive failure: one test registered all twelve images with a low reprojection error, yet the cameras were wrong by 51 percent. The cause was my own bug: some photos carry an EXIF rotation that OpenCV applied but COLMAP ignores. Reprojection error alone would never have caught it; only comparing against ground truth did. For the whole building loop, the image-pair graph falls into disconnected clusters, so incremental reconstruction cannot cross the gaps, and only one side of the building was recovered."

## Block 9 - Advanced work and wrap-up (4:45-5:15)
**On screen:** `results/gpu/bakeoff_inliers.png` (6 s), `results/colmap_compare/overlay_south_12.png` (6 s), closing title.
**Say:**
"For advanced work I added learned features, DISK with LightGlue, on the GPU. They find far more correct matches on hard pairs, for example 379 versus 3,400 under sun flare, and bridge 8 of 13 weak links in the building loop, where SIFT bridged none. But they gave noisier geometry, and on the whole loop more registered cameras did not mean a better model: most were drifted. Compared with COLMAP on identical images, my poses agree to within 0.1 percent. In conclusion: the pipeline reproduces a state-of-the-art result when matches are good, and the failures taught the most: reprojection error can hide a wrong reconstruction. AI tools helped with code and writing; I verified results against ground truth and the declaration is in the report."

---

## Recording checklist
- **Tools:** Windows Game Bar (Win+G) or OBS Studio; 1080p, 30 fps, microphone check first. Record narration separately in Audacity if easier, then sync.
- **Browser/windows:** hide bookmarks and notifications; zoom the browser to 125 % so the HTML viewer is readable.
- **Auto-generated clip (tested, 2026-10-01):** `python scripts/make_demo_video.py --start 40 --count 12 --stride 2 --out results/demo` writes `results/demo/demo_auto.mp4` (58 s, 1280x720, 24 fps, **no audio**, about 1 minute to run). Contents in order: title (0-3 s); the 12 input images (3-10 s); match stages raw -> ratio+mutual -> RANSAC for three image pairs with counts in the title bar (10-27 s); cameras added one by one with a live point count (28-40 s); a 360-degree turntable (40-52 s); reprojection overlays with error histograms (50-58 s). Use it for blocks 1, 2, 3 (growth part) and 5. **Known limits:** it is a sparse-only clip (no dense, no experiments, no failures); the reprojection frames have small text (zoom in your editor); in the turntable the corner cameras are partly cropped by the zoom. It is raw material, not the finished demo: add your voice and the live inspection (block 4).
- **Order of recording:** (1) screen-record block 4 live; (2) assemble the still figures with a free editor (Clipchamp, DaVinci Resolve, or Windows Photos video editor); (3) record voice-over last, reading the script above in your own words.
- **Pace:** about 130 words per minute; the narration above is about 700 words = about 5.5 min, so trim as marked.

## Say it in your own words (the brief tests understanding)
Be ready to answer without notes: *Why four pose candidates and how is one chosen? What does a large reprojection error mean? Why do two-view points have artificially small errors? How are relative poses chained into one frame? What does bundle adjustment minimise? Why did the loop fail?* The answers are in `docs/PROGRESS.md` (sections 4, 6, 8, 9).

## Things NOT to claim (to stay honest and safe under questioning)
- Do not say learned matching "fixed" the loop: it did not (59 registered, 49 wrong). Say the cause of the drift is **not isolated**.
- Do not say our reprojection error is "better than COLMAP": our camera calibration comes from COLMAP and the metrics are not identical.
- Do not present the hall case C or the stride 4-6 learned results as successes: they are examples of silent failure.
- If you add your own photos, show them in a separate block and state the camera and the number of images.

## If you have time for your own photos (strongest extra for "challenging scenarios")
Add 20 seconds after block 8: your own object or building, input photos next to the reconstruction, plus the "tiny step" set showing why too little camera movement ruins depth. See `docs/GUIDE_YOUR_TODO.md`.

## Optional block 10 - your own photographs (strongest "challenging scenarios" material, about 40 s)
**On screen:** `results/own/yellow_all/turntable_object.mp4`, then `results/own/snacks/turntable_object.mp4`, then the negative case `results/own/bottle_classic/turntable_object.mp4` (each about 5 s; compressed H.264 files of 2-8 MB); `docs/report_figs/fig10_own_positive.png` and `fig11_own_negative.png` as stills; the interactive pages `results/viewer/own_yellow_all_object.html`, `own_fire_object_v2.html`, `own_snacks_object.html`, `own_bottle_classic_object.html`.
**Say:** "I also photographed objects with my phone: a warning sign at two heights, a fire extinguisher, snacks, and a transparent bottle on purpose. The phone is uncalibrated, so I estimated its focal length from the photos and validated that method against COLMAP, where the true value is known (0.5 percent). All 330 photos register, the camera paths come out as near-perfect circles, and the sign's text is readable in the 3D cloud. The bottle shows a limit: only the opaque label and cap reconstruct, because the camera sees the background through the clear plastic - registration succeeded, the geometry did not."
**Do not claim a metric accuracy for these sets: there is no ground truth for them.**
