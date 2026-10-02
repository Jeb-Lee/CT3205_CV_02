# Chessboard calibration shoot (about 10 minutes)

Goal: measure your iPhone's true focal length, so the report can compare it with our self-calibrated value (1154 px at 1600 px, i.e. about 4 % above the EXIF value). It is the missing ground truth for your own captures.

## 1. Prepare the board
- Print `docs/calibration/chessboard_10x7_A4.pdf` on **plain matte paper at 100 % / "Actual size"** (NOT "fit to page"). 10 x 7 squares of 25 mm = 9 x 6 inner corners.
- Stick it flat on something rigid (clipboard, hardboard, thick card). It must not curl or bend.
- **Measure one square with a ruler** (measure 5 squares and divide by 5; expect about 25 mm). Write the number down.
- If your objects were shot from far away (more than about 80 cm), an A4 board will look too small. Then show the pattern full-screen on a monitor instead (`docs/calibration/chessboard_10x7_A4_preview.png`, matte screen, no glare) and measure a square on the screen with a ruler.

## 2. Camera settings: copy exactly what you used for the objects
- Photo mode, **1x main lens** (no 0.5x, 2x or 5x, no pinch zoom), **landscape**, same resolution as the object photos (5712 x 4284), JPEG.
- Flash off, Live Photo off, Night mode off, Portrait off. (Settings > Camera > Macro Control off.)
- Distance: **the same distance as when you photographed the objects** (the iPhone's focal length changes slightly with focus distance). The board should fill about 50-70 % of the frame width. +/- 20 % distance is fine.
- Tap the board and **hold to lock AE/AF**; keep the phone steady. Even light, no glare on the paper, no shadows across the board.

## 3. The photos: 25-30, every one with the WHOLE board visible and sharp
1. **5 frontal:** board centred, then moved into each of the 4 corners of the frame (so the board has been seen in every part of the image: this measures the lens distortion).
2. **8 tilted:** tilt the board (or walk around it) about 30-45 degrees left, right, up and down, 2 photos each.
3. **4 near + 4 far:** board filling about 85 % and about 35 % of the frame.
4. **3-4 with the board turned 90 degrees** (phone stays landscape).
Avoid: all-frontal shots, board edges cut off, blur, glare, a bent board, switching lens.

## 4. Hand them over
- Copy the photos **unedited** (no crop, no rotate, no filter) into `Datasets/Calibration/`.
- Tell me: (a) the measured square size in mm, (b) the approximate distance from the phone to the board, (c) if you used a monitor instead of paper.
- Optional (10 more photos): repeat at a clearly different distance, to show how much the focal length changes with focus.

## 5. What happens next
```
python scripts/calibrate_chessboard.py --folder Datasets/Calibration --cols 9 --rows 6 --square_mm <your measurement>
```
This gives the focal length (at full and at 1600 px resolution), the principal point, lens distortion and the reprojection error, and the percentage by which our self-calibration was off. Expect the chessboard result to be accurate to about 0.5 %. If it disagrees with our +4 %, that is reported honestly too.
