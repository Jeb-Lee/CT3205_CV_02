# Your to-do list and photo-capture guide

Deadline: **4 Oct 2026** (start of Week 8). Marks: baseline 40 / experiments 25 / challenging scenarios 5 / demo 10 / advanced 20.

## 1. Things only you can do (the code can't)

| # | Task | Why it earns marks | Effort |
|---|------|--------------------|--------|
| 1 | **Capture your own image set(s)** (see section 2) | Brief says "small object to a building"; own data = strongest "challenging scenarios" + dataset section | 1-2 h |
| 2 | **Record the demo video** (section 4) | 10% is the demo alone | 1 h |
| 3 | **Take screenshots** of the interactive Open3D viewer (`python scripts/view_open3d.py`) | Report figures of a live 3D view are more convincing than static plots | 20 min |
| 4 | **Write the AI Usage Declaration** honestly (template in `docs/AI_DECLARATION.md` once generated) | Mandatory; you are graded on *understanding*, so read every module | 30 min |
| 5 | **Create the private GitHub repo** and push | Mandatory submission item | 15 min |
| 6 | **Read the code so you can explain it** - especially `sfm/geometry.py`, `sfm/reconstruction.py` | The demo checks that you can explain and modify the code | 2 h |

## 2. Photographing your own data

The COLMAP dataset (south-building) is the main dataset. Your own captures add (a) a small object, (b) a building you photographed yourself, and (c) a controlled failure case. Aim for **two sets of ~30-40 photos each**.

### What to shoot (best for points)
1. **A building / monument / statue with lots of unique detail** - brick or stone facade, carved ornament, signage, irregular windows, a fountain, a sculpture. Unique, non-repeating texture is what SIFT loves.
2. **A small textured object on a table** - e.g. a book cover, a plant pot with a patterned cloth, a toy, a shoe, a stone. Put it on a **newspaper/patterned cloth** so the surface is textured too (a bare white table has no features and will break the reconstruction).
3. **One deliberately hard scene** (for "challenging scenarios"): a glass/glossy object, a repetitive facade (identical windows), or a plain wall with one poster. Shoot it the same way and let the pipeline fail - the analysis of *why* is worth marks.

### Avoid
- Glass, mirrors, shiny metal, water, screens (view-dependent reflections).
- Plain walls, sky-only frames, fresh snow, blank paper.
- People, cars, swaying trees, flags (things that move between shots). Foliage is the one weak part of south-building - too much of it hurts.
- Direct sun with hard, moving shadows: **overcast daylight is ideal**.

### How to shoot (this matters more than the phone)
- **Phone main (1x) camera, no zoom, no wide/ultra-wide lens, no portrait mode, no HDR/night mode.** Lock focus/exposure (long-press on the screen). Keep the same lens for every photo, keep it landscape.
- **Move around the object**, don't rotate on the spot. Rotating in place (pure rotation) gives **no depth** - triangulation needs baseline.
- **Step ~10-15 degrees around the object per photo** (about 25-36 photos for a full circle, or a 90-120 degree arc for a building). Neighbouring photos should overlap **70-80%**.
- Keep the **subject distance roughly constant**, shoot from about chest height, and add a **second ring slightly higher** if you can.
- **Keep the phone in LANDSCAPE for every photo.** Portrait shots carry an EXIF rotation tag; our first Person Hall run was silently wrong because of it (see PROGRESS.md section 8.2). Mixing orientations breaks the single-camera-matrix model.
- Keep photos **sharp**: good light, both hands, tap-to-focus. Blur is exactly what Experiment D shows destroys reconstruction.
- **Do not crop, rotate or edit** the photos. Keep EXIF (my code reads the focal length from it).
- Write down: phone model, resolution, number of photos, whether you moved around or rotated the object (required in the report's Dataset section).

### Calibration (optional but earns credibility)
Without calibration the code guesses `f` from EXIF (usually good to ~5%). Better: print a chessboard (the labs used one) and calibrate with `cv2.calibrateCamera`, then save `Datasets/<name>/intrinsics.json` as
`{"fx": ..., "fy": ..., "cx": ..., "cy": ..., "dist": [k1, k2, p1, p2, k3]}`.

### Where to put the files
`Datasets/my-object/images/*.jpg` and `Datasets/my-building/images/*.jpg` (+ optional `intrinsics.json`). Tell me when they are there and I will run the pipeline on them.

## 3. Screenshots worth taking (each is a report figure)
- Open3D window: cloud + cameras from front, top, and a close-up (`scripts/view_open3d.py`).
- The same scene with the **COLMAP** ground-truth cloud overlaid in another colour.
- Any failure (e.g. broken/duplicated geometry) - failure figures are graded under "challenging scenarios".
- Your own photos next to their reconstructions.

## 4. Demo video (about 3-4 minutes)
Suggested storyline (screen recording, e.g. Xbox Game Bar `Win+G`, or OBS):
1. (20 s) Input images strip + dataset description.
2. (40 s) Feature matches: raw -> filtered -> RANSAC (green/red = correct/wrong).
3. (40 s) Estimated cameras appearing one by one (`scripts/make_demo_video.py` will make this clip).
4. (60 s) Live Open3D viewer: rotate/zoom the cloud with cameras.
5. (40 s) Reprojection overlay + error histogram; state numbers.
6. (20 s) Experiments summary + one failure case + COLMAP comparison.
Narrate what each step *means* - that is what the 10% is judged on.
