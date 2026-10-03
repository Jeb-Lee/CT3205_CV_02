"""Build the demo-video kit: per-segment folders of 1920x1080 slides, clips and source media, plus a silent rough cut.

    python scripts/build_demo_kit.py --out "C:/Users/jeble/Claude/CT3205_CV_Assignment_2"

Layout of --out (one folder per block of docs/DEMO_SCRIPT.md):
    00_rough_cut/   block01.mp4 ... block10.mp4 and demo_rough_cut.mp4 (silent, timed to the narration)
    01_intro/ ... 10_advanced/   numbered slides (.png), clips (.mp4) and copied figures, in on-screen order
Numbers on generated slides come from results/*.json|csv; the Person Hall and own-capture tables repeat the report's
Tables VI and VII. Needs ffmpeg with libx264 on PATH.
"""
import argparse, json, os, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--no_video", action="store_true", help="skip clips and the rough cut")
a = ap.parse_args()
OUT = a.out
W, H, FPS = 1920, 1080, 30
BG, FG, MUTED, ACCENT, GOOD, BAD = "#14171c", "#f2f2f2", "#9aa4b2", "#4fc3f7", "#66bb6a", "#ef5350"
FONT_R, FONT_B = "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"
DEMO = "results/demo/demo_auto.mp4"

BLOCKS = {1: ("01_intro", "Intro and dataset"), 2: ("02_matching", "Features and matching"),
          3: ("03_pose", "Camera pose, triangulation, multi-view"), 4: ("04_live_3d", "Live 3D inspection"),
          5: ("05_reprojection", "Reprojection and accuracy"), 6: ("06_dense", "Dense reconstruction"),
          7: ("07_experiments", "Experiments A-D"), 8: ("08_failures", "Failures and challenging scenarios"),
          9: ("09_own_photos", "Own photographs"), 10: ("10_advanced", "Learned features, COLMAP, wrap-up")}
plan = {k: [] for k in BLOCKS}          # rough-cut items per block: (path, seconds or None for a whole clip)


def bdir(k):
    d = os.path.join(OUT, BLOCKS[k][0])
    os.makedirs(d, exist_ok=True)
    return d


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


# ------------------------------------------------------------------------------------------------ slide helpers (PIL)
def trim(s, pad=20):
    """Cut near-white margins (matplotlib figures) so the content fills the slide."""
    g = np.asarray(s.convert("L"))
    ys, xs = np.nonzero(g < 245)
    if len(xs) == 0:
        return s
    return s.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(s.width, xs.max() + pad), min(s.height, ys.max() + pad)))


def slide(k, name, images, title, caption="", stack="h", bg=BG, crops=None):
    """Compose one or more images on a 1920x1080 slide with a block label, title and caption.
    crops: optional per-image (x0, y0, x1, y1) fractions of the source to keep."""
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    d.text((60, 28), f"{k}  ·  {BLOCKS[k][1].upper()}", font=font(26, True), fill=ACCENT)
    d.text((60, 64), title, font=font(46, True), fill=FG)
    top, bottom = 140, H - (90 if caption else 40)
    srcs = []
    for i, p in enumerate(images):
        s = Image.open(p).convert("RGB")
        if crops and crops[i]:
            x0, y0, x1, y1 = crops[i]
            s = s.crop((int(x0 * s.width), int(y0 * s.height), int(x1 * s.width), int(y1 * s.height)))
        srcs.append(trim(s))
    n, gap = len(srcs), 30
    if stack == "h":
        boxes = [(60 + i * ((W - 120 - gap * (n - 1)) // n + gap), top, (W - 120 - gap * (n - 1)) // n, bottom - top) for i in range(n)]
    else:
        hh = (bottom - top - gap * (n - 1)) // n
        boxes = [(60, top + i * (hh + gap), W - 120, hh) for i in range(n)]
    for s, (x, y, bw, bh) in zip(srcs, boxes):
        f = min(bw / s.width, bh / s.height)
        s = s.resize((max(1, int(s.width * f)), max(1, int(s.height * f))), Image.LANCZOS)
        im.paste(s, (x + (bw - s.width) // 2, y + (bh - s.height) // 2))
    if caption:
        d.text((60, H - 70), caption, font=font(30), fill=MUTED)
    p = os.path.join(bdir(k), name)
    im.save(p)
    return p


def fig_slide(k, name, title, caption=""):
    """Start a dark 16:9 matplotlib figure with the block label and title; returns (fig, save)."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(60 / W, 1 - 44 / H, f"{k}  ·  {BLOCKS[k][1].upper()}", color=ACCENT, fontsize=19, weight="bold", va="center")
    fig.text(60 / W, 1 - 100 / H, title, color=FG, fontsize=33, weight="bold", va="center")
    if caption:
        fig.text(60 / W, 50 / H, caption, color=MUTED, fontsize=21, va="center")

    def save():
        p = os.path.join(bdir(k), name)
        fig.savefig(p, facecolor=BG, dpi=100)
        plt.close(fig)
        return p
    return fig, save


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=FG, labelsize=15)
    ax.grid(alpha=0.15, color=FG)
    ax.set_xlabel(xlabel, color=FG, fontsize=16)
    ax.set_ylabel(ylabel, color=FG, fontsize=16)
    ax.set_title(title, color=FG, fontsize=19, weight="bold", loc="left")


def table(ax, header, rows, widths, colors=None, size=22, row_h=None):
    """Plain text table drawn in axes coordinates; colors: per-row text colour or None."""
    ax.axis("off")
    n = len(rows) + 1
    rh = row_h or 0.9 / n           # leave the bottom 10 % free so the last row clears the caption
    xs = np.concatenate([[0], np.cumsum(widths)[:-1]]) / sum(widths)
    for j, h in enumerate(header):
        ax.text(xs[j] + 0.01, 1 - rh / 2, h, color=ACCENT, fontsize=size, weight="bold", va="center")
    ax.plot([0, 1], [1 - rh, 1 - rh], color=MUTED, lw=1.5)
    for i, r in enumerate(rows):
        y = 1 - rh * (i + 1.5)
        c = (colors or [FG] * len(rows))[i]
        for j, v in enumerate(r):
            ax.text(xs[j] + 0.01, y, v, color=c, fontsize=size, va="center", weight="bold" if c != FG else "normal")
        if i < len(rows) - 1:
            ax.plot([0, 1], [y - rh / 2, y - rh / 2], color=MUTED, lw=0.4, alpha=0.5)


def card(k, name, title, lines, sub=""):
    """Text card: title plus bullet lines (key concepts or numbers)."""
    fig, save = fig_slide(k, name, title, sub)
    y = 1 - 230 / H
    for ln in lines:
        big = not ln.startswith("  ")
        fig.text(110 / W, y, ("•  " if big else "     ") + ln.strip(), color=FG if big else MUTED,
                 fontsize=30 if big else 25, va="top")
        y -= (70 if big else 56) / H
    return save()


def copy(k, src, name):
    p = os.path.join(bdir(k), name)
    shutil.copy2(src, p)
    return p


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


VF = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x14171c,fps={FPS},format=yuv420p"
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-an"]


def cut(k, name, t0, t1, src=DEMO):
    p = os.path.join(bdir(k), name)
    if not a.no_video:
        ff("-ss", str(t0), "-to", str(t1), "-i", src, "-vf", VF, *ENC, p)
    return p


def video(k, src, name):
    """Re-encode a clip to 1920x1080 / 30 fps so it drops straight into an editor and the rough cut."""
    p = os.path.join(bdir(k), name)
    if not a.no_video:
        ff("-i", src, "-vf", VF, *ENC, p)
    return p


# ================================================================================================ 1 intro
st = json.load(open("results/baseline/stats.json"))
fig, save = fig_slide(1, "01a_title.png", "")
fig.text(0.5, 0.60, "Multi-View 3D Reconstruction\nand Point Cloud Generation", color=FG, fontsize=60, weight="bold", ha="center", va="center", linespacing=1.3)
fig.text(0.5, 0.40, "Incremental Structure-from-Motion, failure analysis and learned features", color=ACCENT, fontsize=28, ha="center")
fig.text(0.5, 0.30, "CT3205 Computer Vision for AI  ·  Assignment 2", color=MUTED, fontsize=24, ha="center")
fig.text(0.5, 0.24, "[Your Name]  ·  [matric. no.]", color=MUTED, fontsize=24, ha="center")
plan[1].append((save(), 5))
plan[1].append((cut(1, "01b_input_images_clip.mp4", 3, 10), None))
names = [f"P1180{n}.JPG" for n in range(181, 204, 2)]
tiles = []
for n in names:
    t = Image.open(f"Datasets/south-building/images/{n}").convert("RGB")
    t.thumbnail((560, 420))
    c = Image.new("RGB", (560, 420 + 34), BG)
    c.paste(t, ((560 - t.width) // 2, 0))
    ImageDraw.Draw(c).text((4, 422), n, font=font(24), fill=MUTED)
    tiles.append(c)
grid = Image.new("RGB", (4 * 580, 3 * 470), BG)
for i, t in enumerate(tiles):
    grid.paste(t, ((i % 4) * 580, (i // 4) * 470))
tmp = os.path.join(bdir(1), "_grid.png")
grid.save(tmp)
plan[1].append((slide(1, "01c_input_images_grid.png", [tmp], "South-building: 12 input images",
                      "3072×2304 photos, every second frame P1180181–P1180203; scaled to 1600 px and undistorted, K from COLMAP"), 13))
os.remove(tmp)

# ================================================================================================ 2 matching
cnt = json.load(open("docs/report_figs/fig2_counts.json"))
mp = "results/baseline/matches/matches_P1180189_P1180191_"
for key, suf, letter in (("a", "a_raw", "a"), ("b", "b_ratio_mutual", "b"), ("c", "c_ransac", "c")):
    c = cnt[key]
    plan[2].append((slide(2, f"02{letter}_matches_{suf[2:]}.png", [mp + suf + ".jpg"],
                          f"{c['stage'][0].upper() + c['stage'][1:]}: {c['matches']:,} matches, {c['wrong']:,} wrong",
                          "P1180189 ↔ P1180191 · green = correct, red = wrong under COLMAP's epipolar geometry"), 7 if key != "c" else 8))
fig, save = fig_slide(2, "02d_filter_cascade.png", "Each filter removes a different kind of error",
                      "Ratio test: ambiguous appearance · mutual check: one-sided matches · RANSAC on E: geometry")
ax = fig.add_axes([0.2, 0.17, 0.74, 0.62])
lab = ["raw\nnearest neighbour", "ratio test +\nmutual check", "RANSAC on E"]
ok_ = [cnt[k]["matches"] - cnt[k]["wrong"] for k in "abc"]
bad = [cnt[k]["wrong"] for k in "abc"]
ax.barh(lab[::-1], ok_[::-1], color=GOOD, label="correct")
ax.barh(lab[::-1], bad[::-1], left=ok_[::-1], color=BAD, label="wrong")
for i, k in enumerate("cba"):
    ax.text(cnt[k]["matches"] + 80, i, f"{cnt[k]['matches']:,} matches, {cnt[k]['wrong']:,} wrong", color=FG, fontsize=22, va="center")
style(ax, "matches (initial pair)")
ax.tick_params(axis="y", labelsize=22)
ax.set_xlim(0, 11500)
ax.legend(fontsize=18, facecolor=BG, labelcolor=FG, edgecolor=MUTED, loc="lower right")
plan[2].append((save(), 8))
cut(2, "02e_matching_clip_3_pairs.mp4", 10, 28)

# ================================================================================================ 3 pose
fig, save = fig_slide(3, "03a_pipeline.png", "Incremental structure-from-motion",
                      "Every new camera is placed against the existing 3D points, so all poses share one frame and one scale")
ax = fig.add_axes([0.03, 0.12, 0.94, 0.7]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
boxes = [(2, 38, "RootSIFT\nfeatures\n8000 / image"), (22, 38, "Match + verify\nratio, mutual,\nRANSAC on E"),
         (42, 38, "Initial pair\nE → 4 poses,\ncheirality"), (62, 38, "Triangulate\nDLT, depth > 0,\n< 2 px, > 1.5°"),
         (62, 6, "Register next image\nPnP + RANSAC\nagainst 3D points"), (42, 6, "Triangulate\nnew points,\nextend tracks"),
         (22, 6, "Bundle adjustment\nall cameras + points,\nmin reprojection error"), (82, 38, "Sparse cloud\n+ camera poses")]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 16, 16, boxstyle="round,pad=0.6", fc="#1f2630", ec=ACCENT, lw=2))
    ax.text(x + 8, y + 8, t, color=FG, fontsize=19, ha="center", va="center", linespacing=1.4)
arr = dict(arrowstyle="-|>", color=ACCENT, lw=2.5, mutation_scale=25)
for x0, y0, x1, y1 in [(19, 46, 22, 46), (39, 46, 42, 46), (59, 46, 62, 46), (70, 37, 70, 23), (61, 14, 59, 14), (41, 14, 39, 14)]:
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=arr)
ax.annotate("", (66, 23), (28, 23), arrowprops=dict(arr, connectionstyle="arc3,rad=-0.18"))
ax.text(47, 30.5, "repeat until no image can be added", color=MUTED, fontsize=17, ha="center")
ax.annotate("", (82, 46), (79, 46), arrowprops=arr)
plan[3].append((save(), 12))
tv = json.load(open("results/baseline/two_view_demo.json"))
fig, save = fig_slide(3, "03b_cheirality_test.png", "Cheirality test: one of four poses is physically valid",
                      f"Initial pair {tv['pair'][0][:-4]} ↔ {tv['pair'][1][:-4]}: E = [t]× R, SVD → R ∈ {{UWVᵀ, UWᵀVᵀ}}, t = ±u₃")
ax = fig.add_axes([0.06, 0.2, 0.88, 0.54])
rows, cols = [], []
for c in tv["candidates"]:
    sign = "+" if c["t"][0] > 0 else "−"
    rows.append([f"{c['candidate']}", f"{c['rot_angle_deg']:.1f}°", f"{sign}t", f"{c['points_in_front_of_both']:,} of {c['n']:,}",
                 "valid" if c["points_in_front_of_both"] == c["n"] else "points behind a camera"])
    cols.append(GOOD if c["points_in_front_of_both"] == c["n"] else FG)
table(ax, ["#", "Rotation", "Translation", "In front of both cameras", "Verdict"], rows, [0.6, 1.9, 2.1, 4.0, 3.2], cols, size=25)
plan[3].append((save(), 12))
plan[3].append((cut(3, "03c_camera_growth_clip.mp4", 28, 41), None))
plan[3].append((card(3, "03d_key_concepts.png", "How all cameras end up in one coordinate frame", [
    "World frame = first camera:  P₁ = K [ I | 0 ];  first baseline = unit scale",
    "  Monocular SfM is defined only up to scale",
    "Next image: PnP + RANSAC on 2D–3D matches  →  Pᵢ = K [ Rᵢ | tᵢ ]",
    "  Directly in the common frame, with the same scale",
    "Chaining pairwise poses would accumulate error and lose the scale",
    "Bundle adjustment after each image:",
    "  minimise Σ ρ( ‖ π(K, RᵢXⱼ + tᵢ) − xᵢⱼ ‖² )  over all poses and points (Huber, sparse Jacobian)",
]), 13))

# ================================================================================================ 4 live
plan[4].append((card(4, "04a_live_recording_placeholder.png", "LIVE: replace with your screen recording", [
    "Open 04_live_3d/south_building.html in the browser (zoom 125 %)",
    "Front view  →  top-down (L-shaped footprint, camera arc)  →  facade close-up",
    "Toggle camera frusta and the COLMAP overlay; pause 2 s on each view",
    "  Fallback if live fails: the turntable and stills that follow",
]), 5))
shutil.copy2("results/viewer/south_building.html", os.path.join(bdir(4), "south_building.html"))
plan[4].append((video(4, "results/baseline/turntable.mp4", "04b_turntable_fallback.mp4"), None))
plan[4].append((slide(4, "04c_cloud_and_cameras.png", ["results/baseline/scene_views.png"],
                      f"{st['n_points']:,} points, {st['n_registered']} cameras with viewing directions",
                      "Points coloured from the photos; red = camera positions and orientations"), 12))
plan[4].append((slide(4, "04d_plan_and_front_view.png", ["docs/report_figs/fig3_reconstruction.png"],
                      "Plan view: L-shaped footprint with the cameras on an arc", "Report Fig. 3"), 12))

# ================================================================================================ 5 reprojection
plan[5].append((slide(5, "05a_reprojection_check.png", ["results/baseline/reproj_P1180193.jpg"],
                      f"Reprojection check: mean {st['reproj_mean']:.2f} px, median {st['reproj_median']:.2f} px",
                      "Green = detected feature, red = reprojected 3D point; large error → wrong pose, bad point or false match"), 10))
fig, save = fig_slide(5, "05b_accuracy_table.png", "Baseline accuracy (report Table I)",
                      "COLMAP is a reference reconstruction, not ground truth: 'error' means disagreement with COLMAP")
ax = fig.add_axes([0.12, 0.13, 0.76, 0.66])
rows = [["Images registered", f"{st['n_registered']} / {st['n_images']}"],
        ["3D points (seen in ≥ 3 views)", f"{st['n_points']:,} ({st['n_points_track3']:,})"],
        ["Reprojection error, all observations", f"{st['reproj_mean']:.3f} px mean, {st['reproj_median']:.3f} px median"],
        ["Reprojection error, points in ≥ 3 views", f"{st['reproj_mean_track3']:.3f} px mean"],
        ["Camera position vs COLMAP", f"{100 * st['cam_pos_rmse_rel']:.2f} % of the trajectory extent"],
        ["Camera rotation vs COLMAP", f"{st['cam_rot_mean_deg']:.2f}° mean, {st['cam_rot_max_deg']:.2f}° max"],
        ["Points within 2 % of a COLMAP point", f"{100 * st['pt_within_2pct']:.1f} %"]]
table(ax, ["Quantity", "Value"], rows, [1.1, 1], size=25)
plan[5].append((save(), 15))
cut(5, "05c_reprojection_clip.mp4", 47, 56)

# ================================================================================================ 6 dense
plan[6].append((slide(6, "06a_sparse_vs_dense.png", ["results/dense/sparse_vs_dense.png"], "Sparse 18,216 → dense 428,018 points (23.5×)",
                      "16 consecutive images, 15 rectified stereo pairs, semi-global matching, voxel-vote fusion"), 10))
plan[6].append((video(6, "results/dense/dense_turntable.mp4", "06b_dense_turntable.mp4"), None))
plan[6].append((card(6, "06c_dense_numbers.png", "Dense stage: how and how good (report V-F)", [
    "Rectify neighbouring pairs → epipolar lines become rows → SGBM per pixel",
    "Fuse depth maps: keep a voxel only if ≥ 2 stereo pairs put a point in it",
    "COLMAP coverage within 2 %: 58.5 % (sparse) → 66.4 % (dense)",
    "Individual dense points about 35 % noisier than sparse ones",
    "  Support 1 pair: 1.1 M points, 46 % on the facade plane; 3 pairs: 224 k, 80 %",
]), 11))

# ================================================================================================ 7 experiments
A = pd.read_csv("results/expA/results.csv").sort_values("N")
fig, save = fig_slide(7, "07a_expA_number_of_images.png", "A · Number of images: more points, not always more accuracy",
                      "N = 4 registers only 3 cameras; its camera error is unreliable (3 nearly collinear cameras)")
for i, (col, f, lab) in enumerate([("n_points", 1, "3D points"), ("cam_pos_rmse_rel", 100, "camera position error (%)"),
                                   ("reproj_mean", 1, "mean reprojection error (px)"), ("seconds", 1, "runtime (s)")]):
    ax = fig.add_axes([0.06 + (i % 2) * 0.48, 0.12 + (1 - i // 2) * 0.36, 0.42, 0.27])
    ax.plot(A.N, A[col] * f, "o-", color=ACCENT, lw=2.5, ms=9)
    if col == "cam_pos_rmse_rel":
        best = A.loc[A[col].idxmin()]
        ax.plot(best.N, best[col] * f, "o", color=GOOD, ms=16, mfc="none", mew=3)
        ax.annotate(f"best: N = {int(best.N)}", (best.N, best[col] * f), (best.N + 1.5, 0.45), color=GOOD, fontsize=17)
    style(ax, "number of images" if i >= 2 else "", "", lab)
plan[7].append((save(), 9))
B = pd.read_csv("results/expB/results.csv")
fig, save = fig_slide(7, "07b_expB_detectors.png", "B · Detector: RootSIFT best, ORB weakest",
                      "Same 12 images; binary detectors were not tuned")
for i, (col, f, lab) in enumerate([("n_registered", 1, "images registered (of 12)"), ("mean_inliers_per_pair", 1, "verified inliers per pair"),
                                   ("n_points", 1, "3D points"), ("cam_pos_rmse_rel", 100, "camera position error (%)")]):
    ax = fig.add_axes([0.06 + (i % 2) * 0.48, 0.12 + (1 - i // 2) * 0.36, 0.42, 0.27])
    ax.bar(B.label, B[col] * f, color=[GOOD if l == "rootsift" else ACCENT for l in B.label])
    style(ax, "", "", lab)
plan[7].append((save(), 9))
C = pd.read_csv("results/expC/results.csv")
C2 = pd.read_csv("results/expC/two_view_baseline_study.csv")
fig, save = fig_slide(7, "07c_expC_viewpoint.png", "C · Viewpoint change: matching collapses near 35°",
                      "8 images per run; the smallest stride gives the weakest depth (92 % vs 99 % of points within 2 %)")
ax = fig.add_axes([0.06, 0.48, 0.42, 0.27]); ax.bar(C.stride, C.n_registered, color=ACCENT); style(ax, "", "", "images registered (of 8) vs frame stride")
ax = fig.add_axes([0.54, 0.48, 0.42, 0.27]); ax.plot(C.stride, C.mean_inliers_per_pair, "o-", color=ACCENT, lw=2.5, ms=9); style(ax, "", "", "verified inliers per pair vs stride")
ax = fig.add_axes([0.06, 0.12, 0.42, 0.27]); ax.bar(C.stride, 100 * C.pt_within_2pct.fillna(0), color=[BAD if s == 1 else ACCENT for s in C.stride])
for s_, v_ in zip(C.stride, C.pt_within_2pct):
    if np.isnan(v_):
        ax.text(s_, 86, "n/a\n(2 imgs)", color=MUTED, fontsize=14, ha="center")
ax.set_ylim(85, 100); style(ax, "frame stride", "", "% of points within 2 % of COLMAP")
ax = fig.add_axes([0.54, 0.12, 0.42, 0.27])
v = C2.dropna(subset=["median_parallax_deg"])
ax.plot(v.median_parallax_deg, v.inliers, "o-", color=ACCENT, lw=2.5, ms=9)
ax.axvline(35, color=BAD, ls="--", lw=2); ax.text(35.5, v.inliers.max() * 0.8, "≈ 35°: noise\nlevel beyond", color=BAD, fontsize=16)
ax.set_xlim(8, 42); style(ax, "median parallax (°)", "", "two-view inliers vs parallax")
plan[7].append((save(), 8))
D = pd.read_csv("results/expD/results.csv")
fig, save = fig_slide(7, "07d_expD_image_quality.png", "D · Image quality: blur is the most destructive",
                      "Bars: 3D points kept (% of baseline); labels: images registered of 12")
ax = fig.add_axes([0.16, 0.12, 0.38, 0.68])
base = D.loc[D.label == "baseline", "n_points"].iloc[0]
Dd = D[D.label != "baseline"].iloc[::-1]
kept = 100 * Dd.n_points.fillna(0) / base
colors = [BAD if l.startswith("blur") else ACCENT for l in Dd.label]
ax.barh(Dd.label, kept, color=colors)
for y, (kp, r) in enumerate(zip(kept, Dd.n_registered)):
    ax.text(kp + 2, y, f"{int(r)}/12", color=FG, fontsize=14, va="center")
style(ax, "% of baseline points", "", "")
ax.tick_params(axis="y", labelsize=15)
ax2 = fig.add_axes([0.58, 0.12, 0.39, 0.68]); ax2.axis("off")
ax2.imshow(np.asarray(Image.open("results/expD/degradation_examples.jpg").convert("RGB")))
plan[7].append((save(), 9))
copy(7, "results/expD/degradation_examples.jpg", "07e_degradation_examples.jpg")

# ================================================================================================ 8 failures
fig, save = fig_slide(8, "08a_person_hall_cases.png", "Person Hall: every case logged (report Table VI)",
                      "Success rule fixed in advance: ≥ 90 % of images registered and position error < 2 %")
ax = fig.add_axes([0.06, 0.12, 0.88, 0.68])
rows = [["A facade", "reference", "12/12", "0.03 %", "success"], ["B foliage", "occlusion", "2/12", "n/a", "failure"],
        ["C arches + flare", "repetition", "12/12", "0.11 %", "success (after EXIF fix)"], ["D corner turn", "scale change", "11/12", "0.07 %", "success"],
        ["Stride 2–8", "viewpoint", "8/8 (×4)", "≤ 0.75 %", "success"], ["F loop, 55 images", "whole loop", "15/55", "0.66 %", "failure"],
        ["F3 loop, 110 images", "whole loop", "44/110", "0.16 %", "failure (one side)"], ["F3, relaxed thresholds", "connectivity", "45/110", "0.19 %", "failure (one side)"],
        ["F3, learned matching", "whole loop", "59/110", "53 %", "failure (drift)"]]
table(ax, ["Case", "Tests", "Registered", "Cam. err", "Outcome"], rows, [2.6, 1.9, 1.5, 1.3, 2.7],
      [BAD if r[4].startswith("failure") else GOOD for r in rows], size=21)
plan[8].append((save(), 9))
plan[8].append((slide(8, "08b_foliage_matches.png", ["results/person_hall/B/matches/matches_IMG_1045_IMG_1048_c_ransac.jpg"],
                      "Foliage: 74 matches survive, all correct, but too few", "Case B: only 2 of 12 images register; failure by occlusion, not by false matches"), 7))
plan[8].append((slide(8, "08c_exif_silent_failure.png", ["results/person_hall/exif_bug_before_after.png"],
                      "Silent failure: 12/12 registered, 0.6 px error, yet 51 % wrong",
                      "Cause: OpenCV applied an EXIF rotation that COLMAP ignores. Fix: IMREAD_IGNORE_ORIENTATION → 0.11 %"), 9))
plan[8].append((slide(8, "08d_loop_pair_graph.png", ["results/person_hall/F/inlier_matrix.png", "results/person_hall/F3r/inlier_matrix.png"],
                      "Whole loop: the image-pair graph falls apart",
                      "Left: every 6th image, 15 clusters. Right: every 3rd, relaxed thresholds: 3 clusters, still only 45/110 register"), 10))
plan[8].append((video(8, "results/person_hall/F3/turntable.mp4", "08e_loop_one_side_turntable.mp4"), None))
copy(8, "docs/report_figs/fig8_failures.png", "08f_report_fig9_failure_evidence.png")

# ================================================================================================ 9 own photos
cs = []
for nm in ("yellow_level", "fire_ext", "snacks", "bottle"):
    cs.append(f"results/own/contact_{nm}.jpg")
g = Image.new("RGB", (3620, 1820), BG)
for i, p in enumerate(cs):
    t = Image.open(p).convert("RGB")
    g.paste(t, ((i % 2) * 1820, (i // 2) * 920))
tmp = os.path.join(bdir(9), "_contacts.png"); g.save(tmp)
plan[9].append((slide(9, "09a_own_input_photos.png", [tmp], "Own photos: four objects, 330 iPhone photos, one orbit each",
                      "Yellow sign (two heights), fire extinguisher, snacks, and a transparent bottle chosen as a negative case"), 6))
os.remove(tmp)
cb =json.load(open("results/own/chessboard_calibration.json"))
fig, save = fig_slide(9, "09b_focal_length_calibration.png", "Uncalibrated phone: focal length from the photos",
                      "Self-calibration: for the right K, E = KᵀFK has two equal singular values")
ax = fig.add_axes([0.06, 0.14, 0.44, 0.62]); ax.axis("off")
ax.imshow(np.asarray(Image.open("results/own/focal_south_building.png").convert("RGB")))
ax.set_title("validation on south-building (known focal length)", color=FG, fontsize=18, loc="left")
lines = ["Method check, south-building:", "  1326.5 vs COLMAP 1333.2 px  (0.5 %)",
         "Own photos: 1154 px at 1600 px", "  EXIF value × 1.04",
         f"Chessboard, {cb['n_used']} photos: {cb['f_at_working_res']:.0f} px", f"  within {abs(cb['self_calibration_error_pct']):.1f} %; quoted as about ±1 %",
         "Checks the focal length,", "  not the 3D models"]
y = 0.74
for t in lines:
    sub = t.startswith("  ")
    fig.text(0.54, y, ("     " if sub else "•  ") + t.strip(), color=MUTED if sub else FG, fontsize=23 if sub else 27, va="top")
    y -= 0.055 if not sub else 0.08
plan[9].append((save(), 7))
plan[9].append((video(9, "results/own/yellow_all/turntable_object.mp4", "09c_yellow_sign_turntable.mp4"), None))
plan[9].append((video(9, "results/own/snacks/turntable_object.mp4", "09d_snacks_turntable.mp4"), None))
plan[9].append((video(9, "results/own/bottle_classic/turntable_object.mp4", "09e_bottle_negative_turntable.mp4"), None))
fig, save = fig_slide(9, "09f_own_results_table.png", "All 330 photos register; the bottle keeps only its opaque parts",
                      "Orbits: 369–392° coverage, 5–9 % radius spread · no ground truth, so no metric accuracy is claimed")
ax = fig.add_axes([0.06, 0.14, 0.88, 0.62])
rows = [["Yellow sign (2 heights)", "119/119", "28,131", "0.59", "322,793"], ["Fire extinguisher", "83/83", "22,995", "0.43", "576,819"],
        ["Snacks", "71/71", "44,621", "0.42", "1,033,713"], ["Bottle, classical", "57/57", "14,254", "0.38", "35,330 (label, cap)"],
        ["Bottle, DISK + LightGlue", "57/57", "67,129", "0.57", "25,188 (no better)"]]
table(ax, ["Set", "Registered", "Sparse pts", "Reproj. px", "Dense pts (object)"], rows, [3.3, 1.5, 1.5, 1.4, 2.6],
      [FG, FG, FG, BAD, BAD], size=22)
plan[9].append((save(), 7))
video(9, "results/own/fire/turntable_object.mp4", "09g_fire_extinguisher_turntable.mp4")
video(9, "results/own/bottle_learned/turntable_object.mp4", "09h_bottle_learned_turntable.mp4")
copy(9, "docs/report_figs/fig10_own_positive.png", "09i_report_fig10_positive.png")
copy(9, "docs/report_figs/fig11_own_negative.png", "09j_report_fig11_negative.png")
copy(9, "results/own/calib_photo_review.jpg", "09k_chessboard_photos.jpg")
os.makedirs(os.path.join(bdir(9), "viewers"), exist_ok=True)
for v_ in ("own_yellow_all_object", "own_fire_object_v2", "own_snacks_object", "own_bottle_classic_object", "own_bottle_learned_object"):
    shutil.copy2(f"results/viewer/{v_}.html", os.path.join(bdir(9), "viewers", v_ + ".html"))

# ================================================================================================ 10 advanced
plan[10].append((slide(10, "10a_matcher_bakeoff.png", ["results/gpu/bakeoff_inliers.png"], "Learned matching on hard pairs (DISK + LightGlue, GPU)",
                       "Correct inliers (judged with COLMAP's geometry), sun flare: 379 RootSIFT → 3,400 LightGlue · loop links bridged: 0 → 8 of 13",
                       crops=[(0, 0.075, 1, 1)]), 10))
plan[10].append((slide(10, "10b_learned_loop_drift.png", ["docs/report_figs/fig8_failures.png"], "But: 59 cameras on the loop, 49 wrong",
                       "Rotation error grows smoothly along the chain (green = correct): drift without loop closure, cause not isolated",
                       crops=[(0.5, 0.38, 1, 1)]), 9))
plan[10].append((slide(10, "10c_vs_colmap_overlay.png", ["results/colmap_compare/overlay_south_12.png"], "Ours (red) vs COLMAP (blue) on identical images",
                       "Sim(3)-aligned plan view, south-building, 12 images"), 10))
fig, save = fig_slide(10, "10d_colmap_comparison.png", "Agreement with COLMAP (report Table V)",
                      "Lower reprojection error is not superiority: our intrinsics are COLMAP's and its model is constrained by all images")
ax = fig.add_axes([0.08, 0.14, 0.84, 0.62])
rows = [["south-building, 12", "10,934", "8,830", "0.11 %"], ["south-building, 12, DISK + LG", "20,084", "8,830", "0.20 %"],
        ["south-building, 16", "18,216", "11,577", "0.05 %"], ["Person Hall A", "12,584", "17,780", "0.03 %"], ["Person Hall C", "6,509", "10,777", "0.11 %"]]
table(ax, ["Images", "Points (ours)", "Points (COLMAP)", "Camera error"], rows, [3.4, 2, 2.3, 1.8], size=25)
plan[10].append((save(), 8))
fig, save = fig_slide(10, "10e_closing.png", "")
fig.text(0.5, 0.70, "Conclusions", color=FG, fontsize=54, weight="bold", ha="center")
for i, t in enumerate(["Matches COLMAP's camera poses to about 0.1 % when matches are good",
                       "More images, more features or learned matching are not automatically better",
                       "Reprojection error and registration counts can hide a wrong reconstruction",
                       "Transparent surfaces break matching: geometry only where the object is opaque"]):
    fig.text(0.5, 0.56 - i * 0.08, t, color=FG if i != 2 else ACCENT, fontsize=29, ha="center", weight="bold" if i == 2 else "normal")
fig.text(0.5, 0.14, "AI usage is declared in the report  ·  Code: github.com/Jeb-Lee/CT3205_CV_02 (private)", color=MUTED, fontsize=20, ha="center")
plan[10].append((save(), 8))
copy(10, "docs/report_figs/fig9_learned_colmap.png", "10f_report_fig8_learned_colmap.png")

# ================================================================================================ section cards
for k, (folder, title) in BLOCKS.items():
    fig, save = fig_slide(k, "00_section_card.png", "")
    fig.text(0.5, 0.55, f"{k}", color=ACCENT, fontsize=110, weight="bold", ha="center", va="center")
    fig.text(0.5, 0.38, title, color=FG, fontsize=48, weight="bold", ha="center", va="center")
    save()

# ================================================================================================ rough cut
if not a.no_video:
    rc = os.path.join(OUT, "00_rough_cut"); os.makedirs(rc, exist_ok=True)
    tmpd = os.path.join(rc, "_parts"); os.makedirs(tmpd, exist_ok=True)
    blocks = []
    for k, items in plan.items():
        parts = []
        for i, (p, sec) in enumerate(items):
            q = os.path.join(tmpd, f"b{k:02d}_{i:02d}.mp4")
            if sec is None:
                shutil.copy2(p, q)
            else:
                ff("-loop", "1", "-t", str(sec), "-i", p, "-vf", VF, *ENC, q)
            parts.append(q)
        lst = os.path.join(tmpd, f"b{k:02d}.txt")
        open(lst, "w").write("".join(f"file '{os.path.abspath(q)}'\n" for q in parts))
        out = os.path.join(rc, f"block{k:02d}_{BLOCKS[k][0][3:]}.mp4")
        ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out)
        blocks.append(out)
        print(f"block {k}: {len(parts)} items", flush=True)
    lst = os.path.join(tmpd, "all.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(b)}'\n" for b in blocks))
    ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", os.path.join(rc, "demo_rough_cut.mp4"))
    shutil.rmtree(tmpd)
json.dump({BLOCKS[k][0]: [[os.path.relpath(p, OUT).replace(os.sep, "/"), s] for p, s in v] for k, v in plan.items()},
          open(os.path.join(OUT, "00_rough_cut", "rough_cut_plan.json") if not a.no_video else os.path.join(OUT, "rough_cut_plan.json"), "w"), indent=1)
print("kit written to", OUT)
