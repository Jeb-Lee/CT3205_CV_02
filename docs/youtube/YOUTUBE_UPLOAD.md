# YouTube upload fields - CT3205 Assignment 2 demo

Copy each field into YouTube Studio (Create → Upload videos). **Check the chapter times against your final edit before publishing.**

---

## Title (max 100 characters)
```
Computer Vision Assignment 2 – Multi-View 3D Reconstruction with Structure-from-Motion
```
(86 characters.) A shorter alternative: `Computer Vision Assignment 2 – Photos to 3D Point Cloud` (55).

## Description
```
From photos to a 3D point cloud: an incremental Structure-from-Motion pipeline implemented from the geometry up, for CT3205 Computer Vision for AI, Assignment 2.

The pipeline finds RootSIFT features, matches and verifies them, recovers camera poses from the essential matrix (with the cheirality test), triangulates 3D points, registers new images with PnP and refines everything with bundle adjustment. A rectified stereo stage then densifies the cloud.

Highlights
• South-building: 12 images reconstruct to 10,934 points; camera positions agree with COLMAP's model to 0.11% of the trajectory and rotations to 0.18°
• Dense stereo: 18 thousand → 428 thousand points (23×)
• Four controlled experiments: number of images, feature detector, viewpoint change, image quality
• Failure analysis on Person Hall, including a "silent failure": all images registered with 0.6 px error, yet the model was 51% wrong because of an EXIF orientation bug
• Learned matching (DISK + LightGlue on the GPU) and a comparison with COLMAP
• My own iPhone captures of four objects (330 photos), with the focal length self-calibrated and checked against a chessboard; a transparent bottle as a deliberate negative case

Chapters
0:00 Intro and dataset
0:25 Features and matching
0:55 Camera pose, triangulation, multi-view
1:45 Interactive 3D inspection
2:20 Reprojection and accuracy
2:45 Dense reconstruction
3:10 Experiments A–D
3:45 Failures and challenging scenarios
4:25 Own photographs
5:00 Learned features, COLMAP and conclusions

Datasets
South Building and Person Hall from the COLMAP dataset collection (J. L. Schönberger): https://demuc.de/colmap/datasets/

Tools
Python, OpenCV, NumPy, SciPy, Open3D, PyTorch and Kornia (DISK, LightGlue). COLMAP's models are used only as a reference for evaluation.

AI usage is declared in the accompanying report.

Lee Tze Hian · CT3205 Computer Vision for AI
```

## Tags (YouTube allows 500 characters in total)
```
computer vision, structure from motion, SfM, 3D reconstruction, multi-view geometry, point cloud, photogrammetry, bundle adjustment, essential matrix, triangulation, PnP, SIFT, RootSIFT, LightGlue, DISK, dense stereo, SGBM, COLMAP, OpenCV, Python, camera calibration, CT3205
```

## Other settings
| Field | Recommended value | Why |
|---|---|---|
| Thumbnail | `thumbnail_final_1280x720.jpg` (your final design; `thumbnail_final_original.jpg` is the 2000×1116 source) | exactly 1280×720, under 2 MB |
| Playlist | none, or a "Coursework" playlist | optional |
| Audience | **No, it's not made for kids** | coursework, technical content |
| Age restriction | No | |
| Altered or synthetic content | **No** | the video shows real photos and real reconstructions; AI helped with code and writing, not with generating realistic footage |
| Paid promotion | No | |
| Automatic chapters | Off (your chapters in the description take priority) | |
| Featured places / moments | Off | |
| Language | English | also set the caption language to English so auto-captions are generated |
| Captions | Auto-generated, then check the technical words (RootSIFT, cheirality, PnP, COLMAP) | markers may watch with captions |
| Recording date / location | optional | |
| License | Standard YouTube License | |
| Category | **Education** (or Science & Technology) | |
| Comments | Off, or "hold all for review" | avoids spam on coursework |
| Show likes count | your choice | |
| End screen / cards | none | keeps the video clean for markers |
| **Visibility** | **Unlisted** | anyone with the link can watch, but it isn't searchable; the usual choice for submitted coursework |

## Before you publish: check these
- **Chapter times.** YouTube needs the first chapter at 0:00, at least three chapters, and each at least 10 seconds long. Update the times if you cut block 6 or changed any timings. Without block 6, every chapter from "Experiments A–D" onwards moves 25 s earlier: 2:45, 3:20, 4:00 and 4:35.
- **Visibility and your matric number.** The title slide shows your matric number. Keep the video **Unlisted** (or check whether your course wants it Private and shared with specific accounts). If you ever make it Public, consider removing the matric number from the title slide.
- **The course's rules.** Check whether the assignment specifies a hosting platform, visibility or a naming format for the video.
- **Music.** If you add any, use YouTube Audio Library or other royalty-free tracks, to avoid copyright claims.
- **Code link.** The GitHub repository is private, so I haven't put a link in the description. Add it only if you make the repository visible to the markers.
