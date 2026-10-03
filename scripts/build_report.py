"""Build the IEEE two-column report draft from docs/IEEE 2 col conference-template-a4.docx.

    python scripts/make_report_figures.py     # figures first
    python scripts/build_report.py            # -> docs/Report_CT3205_A2_DRAFT.docx

The supplied template is saved in Strict OOXML, which python-docx cannot read; it is converted (namespace rewrite) to a
temporary Transitional copy first. Headings, captions, table heads and references are auto-numbered by the template styles.
Yellow-highlighted text = placeholders ([YOU]) that only the author can fill.
"""
import os
import re
import tempfile
import zipfile

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE = os.path.join(ROOT, "docs", "IEEE 2 col conference-template-a4.docx")
OUT = os.path.join(ROOT, "docs", "Report_CT3205_A2_DRAFT.docx")
FIGS = os.path.join(ROOT, "docs", "report_figs")
COLW = 8.5  # cm, one column


# ------------------------------------------------------------------------------------ template conversion
def strict_to_transitional(src, dst):
    MAP = [("http://purl.oclc.org/ooxml/wordprocessingml/main", "http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
           ("http://purl.oclc.org/ooxml/officeDocument/relationships", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
           ("http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
           ("http://purl.oclc.org/ooxml/drawingml/main", "http://schemas.openxmlformats.org/drawingml/2006/main"),
           ("http://purl.oclc.org/ooxml/drawingml/picture", "http://schemas.openxmlformats.org/drawingml/2006/picture"),
           ("http://purl.oclc.org/ooxml/officeDocument/math", "http://schemas.openxmlformats.org/officeDocument/2006/math"),
           ("http://purl.oclc.org/ooxml/officeDocument/extendedProperties", "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"),
           ("http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes", "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"),
           ("http://purl.oclc.org/ooxml/officeDocument/sharedTypes", "http://schemas.openxmlformats.org/officeDocument/2006/sharedTypes"),
           ("http://purl.oclc.org/ooxml/officeDocument/customXml", "http://schemas.openxmlformats.org/officeDocument/2006/customXml")]
    zin = zipfile.ZipFile(src)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith((".xml", ".rels")):
                t = data.decode("utf-8")
                for a, b in MAP:
                    t = t.replace(a, b)
                t = t.replace("relationships/extendedProperties", "relationships/extended-properties")
                data = t.encode("utf-8")
            zout.writestr(item, data)


tmp = os.path.join(tempfile.gettempdir(), "ieee_template_transitional.docx")
strict_to_transitional(TEMPLATE, tmp)
doc = docx.Document(tmp)
paras = list(doc.paragraphs)
P_TITLE, P_ABS, P_KEY, ANCHOR = paras[0], paras[10], paras[11], paras[89]

# ------------------------------------------------------------------------------------ remove template guidance text
for p in paras[12:89]:
    p._p.getparent().remove(p._p)
for t in list(doc.tables):
    t._tbl.getparent().remove(t._tbl)
for k in (1, 2, 5, 6, 7):                          # sub-title note, spacer, extra authors
    paras[k]._p.getparent().remove(paras[k]._p)
# author block: one centred author instead of three columns. ONLY the first 3-column section is changed: also changing the
# empty spacer section after it makes Word start the abstract on a new page (tested).
for sect in doc.element.body.iter(qn("w:sectPr")):
    cols = sect.find(qn("w:cols"))
    if cols is not None and cols.get(qn("w:num")) == "3":
        cols.set(qn("w:num"), "1")
        break


# ------------------------------------------------------------------------------------ helpers
TOKEN = re.compile(r"(</?(?:b|i|hl|sub|sup)>)")


def add_runs(p, text):
    st = dict(b=False, i=False, hl=False, sub=False, sup=False)
    for tok in TOKEN.split(text):
        m = re.fullmatch(r"<(/?)(b|i|hl|sub|sup)>", tok)
        if m:
            st[m.group(2)] = m.group(1) == ""
            continue
        if not tok:
            continue
        r = p.add_run(tok)
        r.bold = st["b"] or None
        r.italic = st["i"] or None
        if st["hl"]:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if st["sub"]:
            r.font.subscript = True
        if st["sup"]:
            r.font.superscript = True
    return p


def new_p(text="", style="Body Text"):
    p = doc.add_paragraph(style=style)
    add_runs(p, text)
    ANCHOR._p.addprevious(p._p)
    return p


def H1(t): return new_p(t, "Heading 1")
def H2(t): return new_p(t, "Heading 2")
def B(t): return new_p(t, "Body Text")
def BUL(t): return new_p(t, "bullet list")


def EQ(expr, num):
    p = doc.add_paragraph(style="equation")
    p.add_run("\t")
    add_runs(p, expr)
    p.add_run("\t(%d)" % num)
    for r in p.runs:
        r.font.name = "Times New Roman"
        rf = r._r.get_or_add_rPr().find(qn("w:rFonts"))
        for a in ("w:cs", "w:eastAsia"):
            rf.set(qn(a), "Times New Roman")
    ANCHOR._p.addprevious(p._p)


def FIG(fname, caption):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(os.path.join(FIGS, fname), width=Cm(COLW))
    ANCHOR._p.addprevious(p._p)
    c = doc.add_paragraph(style="figure caption")
    add_runs(c, caption)
    ANCHOR._p.addprevious(c._p)


def set_borders(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcPr.append(b)
    for edge, val in kw.items():
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single" if val else "nil")
        if val:
            e.set(qn("w:sz"), str(val)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)


def TABLE(caption, header, rows, widths, note=None, left_cols=(0,)):
    cap = doc.add_paragraph(style="table head")
    add_runs(cap, caption)
    cap.paragraph_format.keep_with_next = True
    ANCHOR._p.addprevious(cap._p)
    tw = sum(widths)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement("w:" + side); e.set(qn("w:w"), "45"); e.set(qn("w:type"), "dxa"); mar.append(e)
    tblPr.append(mar)
    for ri, rowdata in enumerate([header] + rows):
        for ci, val in enumerate(rowdata):
            cell = t.cell(ri, ci)
            cell.width = Cm(widths[ci] * COLW / tw)
            para = cell.paragraphs[0]
            para.style = doc.styles["table col head" if ri == 0 else "table copy"]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci in left_cols else WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.keep_with_next = ri < len(rows)     # keep the whole table together
            add_runs(para, str(val))
            set_borders(cell, top=(6 if ri == 0 else 0), bottom=(6 if ri in (0, len(rows)) else 0))
    ANCHOR._p.addprevious(t._tbl)
    if note:
        n = doc.add_paragraph(style="table copy")      # not 'table footnote': that style auto-letters (a., b., c.)
        n.paragraph_format.space_before = Pt(2)
        n.paragraph_format.space_after = Pt(4)
        add_runs(n, note)
        ANCHOR._p.addprevious(n._p)
    else:
        sp = doc.add_paragraph(style="Normal"); sp.paragraph_format.space_after = Pt(2)
        ANCHOR._p.addprevious(sp._p)


def REF(text):
    p = doc.add_paragraph(style="references")
    add_runs(p, text)
    ANCHOR._p.addprevious(p._p)


# ------------------------------------------------------------------------------------ front matter
for r in list(P_TITLE.runs):
    r._r.getparent().remove(r._r)
P_TITLE.add_run("Multi-View 3D Reconstruction and Point Cloud Generation: Incremental Structure-from-Motion, "
                "Failure Analysis and Learned-Feature Extensions")
author = [p for p in doc.paragraphs if p.style.name == "Author" and p.text.strip().startswith("line 1")][0]
for ch in list(author._p):
    if ch.tag != qn("w:pPr"):
        author._p.remove(ch)
r = author.add_run("[Your Name]"); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
for line in ("College of Computing and Data Science", "Nanyang Technological University, Singapore",
             "CT3205 Computer Vision for AI, Assignment 2", "[email]  |  matric. no. [XXXXXXX]"):
    r.add_break()
    r2 = author.add_run(line); r2.italic = True
    if line.startswith("["):
        r2.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r = r2

for ch in list(P_ABS._p):
    if ch.tag != qn("w:pPr"):
        P_ABS._p.remove(ch)
a0 = P_ABS.add_run("Abstract"); a0.italic = True
P_ABS.add_run("—We implement an incremental multi-view 3D reconstruction pipeline from the geometry up: feature matching with "
              "geometric verification, essential-matrix pose recovery with the cheirality test, linear triangulation, PnP-based "
              "registration into a common frame, sparse bundle adjustment and reprojection checks. Against the COLMAP model of the "
              "dataset, twelve south-building images reconstruct to 10,934 points with camera positions within 0.11% of the "
              "trajectory extent and rotations within 0.18°. Four controlled experiments vary the number of images, the detector, "
              "the viewpoint change and the image quality: more images do not always improve accuracy, RootSIFT outperforms SIFT, ORB, "
              "AKAZE and BRISK, matching collapses beyond about 35° of viewpoint change, and blur is the most destructive "
              "degradation. We log every success and failure, including a reconstruction that registered all images with low "
              "reprojection error yet was 51% wrong because of an EXIF-orientation bug, and a building loop that fragments into "
              "disconnected image clusters. Advanced work adds a rectified-SGBM dense stage (23× more points), GPU learned matching "
              "with DISK and LightGlue, and a comparison with COLMAP. Five own iPhone captures (330 photos), self-calibrated from "
              "fundamental matrices (validated to 0.5%), reconstruct completely; a transparent bottle, chosen as a negative case, "
              "yields geometry only where it is opaque. The main lesson is that reprojection error and registration counts can hide "
              "a wrong or incomplete reconstruction.")
for ch in list(P_KEY._p):
    if ch.tag != qn("w:pPr"):
        P_KEY._p.remove(ch)
k0 = P_KEY.add_run("Keywords"); k0.italic = True
P_KEY.add_run("—structure from motion, multi-view geometry, essential matrix, bundle adjustment, dense stereo, learned local features, failure analysis")

for ch in list(ANCHOR._p):
    if ch.tag != qn("w:pPr"):
        ANCHOR._p.remove(ch)
ANCHOR.style = doc.styles["Normal"]

# ======================================================================================== I. INTRODUCTION
H1("Introduction")
B("Recovering the 3D structure of a static scene and the positions of the cameras that photographed it, from overlapping "
  "images alone, is the core problem of structure from motion (SfM) [1], [8]. In the stereo laboratory a scene was reconstructed "
  "from a pair of images by estimating the relative camera motion from feature correspondences and triangulating. This work "
  "extends that pipeline to many views: it must produce feature correspondences, relative camera poses, 3D points, a sparse "
  "point cloud, camera positions and orientations, and a reprojection of the points into the input images.")
B("The aim is not an attractive point cloud but an understanding of each stage and of how reconstruction quality depends on the "
  "images. We therefore implemented the pipeline ourselves instead of calling an existing SfM package, evaluated it quantitatively "
  "against the COLMAP model shipped with the datasets, and investigated how the number and arrangement of images, the feature "
  "detector and the image quality affect the result. Our contributions are:")
BUL("an incremental SfM pipeline with explicit essential-matrix decomposition and cheirality test, DLT triangulation, PnP "
    "registration, track management and sparse bundle adjustment (Section III);")
BUL("four controlled experiments and a quantitative comparison with COLMAP (Section V);")
BUL("a fully logged study of challenging scenarios including failures and a diagnosed implementation bug (Section VI);")
BUL("advanced work: dense stereo reconstruction, GPU learned-feature matching (DISK + LightGlue) and an analysis of what COLMAP "
    "does that our pipeline does not (Sections III-I, III-J and V).")
B("Main findings: the pipeline agrees with COLMAP to about 0.1% in camera position; accuracy is not monotonic in the number of "
  "images; matching collapses near 35° of viewpoint change; and several failures were silent—the reconstruction looked "
  "successful by reprojection error and registration count while being wrong.")

# ======================================================================================== II. DATASETS
H1("Datasets")
H2("South-building (main dataset)")
B("From the COLMAP dataset collection [15]: 128 photographs (3072×2304) of a brick building taken from the ground, with a "
  "SIMPLE_RADIAL camera (f = 2559.7 px, k<sub>1</sub> = −0.0205). The baseline and control sequence uses 12 images "
  "(P1180181–P1180203, every second frame); the dense reconstruction uses 16 consecutive images. Images are scaled to 1600 px on "
  "the long side and undistorted, and the camera matrix is scaled accordingly. Consecutive frames have about 12° of parallax and "
  "matching survives up to about 35° (Section V-D).")
H2("Person-hall (challenging dataset)")
B("330 photographs (5616×3744, OPENCV model, k<sub>1</sub> = −0.109) of a walk around a brick building that returns to its "
  "start (Fig. 1d–f). It contains heavy foliage, repetitive arched windows, sun flare, strong scale changes and 48 portrait "
  "photographs carrying an EXIF orientation tag (Section VI).")
H2("Own photographs")
B("Five sets (330 photographs) were taken by hand with an iPhone 16 Pro Max main camera (24 mm-equivalent, f/1.78, 1/100 s, "
  "ISO 80–125, JPEG, 5712×4284; 10 frames at 4032×3024 with the same field of view). The camera was walked once round "
  "a stationary object standing on a pedestal in a textured office, at roughly constant distance, without turning the object: a "
  "yellow warning sign at two heights (82 + 37 photos), a fire extinguisher (83), snacks (71) and a transparent bottle (57), the "
  "last chosen on purpose as a negative case (Fig. 10–11, Section VI). <hl>[YOU] confirm: shooting distance, camera heights "
  "and lighting.</hl>")
H2("Ground truth and its limits")
B("The COLMAP sparse models shipped with the datasets are used only for the camera intrinsics and for evaluation, never as input to "
  "our reconstruction. They are reconstructions themselves, so “error” below means disagreement with COLMAP, and our intrinsics "
  "are COLMAP’s rather than self-calibrated.")
FIG("fig1_datasets.png", "Representative input images. (a)–(c) south-building; (d) Person Hall clean facade, (e) foliage, "
    "(f) arched windows with sun flare.")

# ======================================================================================== III. METHODOLOGY
H1("Methodology")
H2("Feature detection and description")
B("We use RootSIFT [2], [3] (SIFT with L<sub>1</sub>-normalised, square-rooted descriptors), detecting 8000 keypoints per image with "
  "a contrast threshold of 0.02. SIFT, ORB, AKAZE and BRISK (OpenCV [13]) are compared in Experiment B and learned DISK features in "
  "Section III-J.")
H2("Matching and filtering")
B("Matches pass a cascade (Fig. 2): raw nearest-neighbour matching; Lowe’s ratio test (0.8); a mutual (cross-check) test; and "
  "geometric verification by RANSAC on the essential matrix [5] with a 1 px threshold. We judge correctness against the COLMAP "
  "epipolar geometry (Sampson distance < 1.5 px): green lines are correct, red are wrong. For the initial pair, 8000 raw matches "
  "(4,467 wrong) reduce to 3164 after the ratio and mutual tests (81 wrong) and 3077 after RANSAC (none wrong).")
FIG("fig2_matching.png", "Match filtering for the initial pair (P1180189, P1180191): (a) raw nearest neighbour, (b) ratio test and "
    "mutual check, (c) after RANSAC. Green = correct, red = wrong under COLMAP geometry. 70 randomly chosen matches drawn.")
H2("Essential matrix and relative camera pose")
B("For normalised image points x<sub>1</sub>, x<sub>2</sub> of the same 3D point, the essential matrix satisfies")
EQ("<i>E</i> = [<i>t</i>]<sub>×</sub> <i>R</i>,     <i>x</i><sub>2</sub><sup>T</sup> <i>E x</i><sub>1</sub> = 0", 1)
B("and is estimated from the verified matches and the camera matrix K. Its SVD gives four candidate poses, "
  "<i>R</i> ∈ {<i>UWV</i><sup>T</sup>, <i>UW</i><sup>T</sup><i>V</i><sup>T</sup>} combined with <i>t</i> = ±<i>u</i><sub>3</sub>, "
  "so the pose is ambiguous. The physically valid solution is selected by the <i>cheirality</i> condition: the triangulated points "
  "must lie in front of both cameras. On the initial pair the four candidates place 0, 3077, 0 and 0 points in front of both "
  "cameras, so exactly one is valid; our rotation agrees with OpenCV’s recoverPose to 0.0°.")
H2("Multi-view camera pose estimation")
B("<b>Common frame.</b> The first camera of the initial pair defines the world, <i>P</i><sub>1</sub> = <i>K</i>[<i>I</i> | 0], and the "
  "baseline of that pair defines unit scale (monocular reconstruction is defined only up to scale). The initial pair is chosen among "
  "the best-connected candidates by the number of valid triangulated points weighted by median parallax (at least 3°), which "
  "avoids near-pure-rotation pairs whose depth is ill-conditioned.")
B("<b>Registering images.</b> Matches whose partner keypoint already belongs to a 3D point give 2D–3D correspondences. The "
  "image with the most correspondences is registered by PnP with RANSAC (EPnP minimal sets [6], 3 px threshold) followed by "
  "Levenberg–Marquardt refinement, which yields <i>P</i><sub>i</sub> = <i>K</i>[<i>R</i><sub>i</sub> | <i>t</i><sub>i</sub>] "
  "directly in the common frame. New points are then triangulated between the new image and every registered image, and existing "
  "tracks are extended.")
B("<b>Why PnP instead of chaining relative poses.</b> Chaining pairwise transforms, <i>R</i><sub>i</sub> = "
  "<i>R</i><sub>i,i−1</sub><i>R</i><sub>i−1</sub>, accumulates error and leaves the translation scale of every pair "
  "undetermined. PnP against the shared 3D points inherits a single consistent scale and ties every camera to the same structure. "
  "A <i>track</i> stores, for each 3D point, the keypoint that observes it in each image; it links images and supports PnP, "
  "triangulation and bundle adjustment. After each registration, bundle adjustment (III-F) refines everything and observations "
  "with error above 2 px or negative depth are dropped. Fig. 3 shows the result: twelve cameras on an arc around the building.")
FIG("fig3_reconstruction.png", "Baseline reconstruction (12 images, 10,934 points): (a) plan view and (b) front view with camera "
    "positions and viewing directions (red arrows). The scene is levelled using the cameras’ mean up-axis.")
H2("Triangulation")
B("For a correspondence with projection matrices <i>P</i><sub>1</sub>, <i>P</i><sub>2</sub> (in normalised coordinates), each view "
  "contributes two linear equations on <i>X</i>, with rows <i>x p</i><sub>3</sub> − <i>p</i><sub>1</sub> and "
  "<i>y p</i><sub>3</sub> − <i>p</i><sub>2</sub> where <i>p</i><sub>k</sub> are the rows of <i>P</i>. The four stacked rows give "
  "<i>AX</i> = 0, solved by the singular vector of the smallest singular value (DLT [1]); geometrically the projection matrices "
  "define two viewing rays and <i>X</i> is their intersection. A point is accepted only if its depth is positive in both cameras, "
  "its reprojection error is below 2 px and its triangulation angle exceeds 1.5° (small angles give unreliable depth). Our DLT "
  "agrees with OpenCV’s triangulatePoints to 8×10<sup>−12</sup>.")
H2("Bundle adjustment")
B("We minimise the robust reprojection error over all camera poses (the first camera fixed) and all 3D points,")
EQ("min Σ<sub>ij</sub> ρ( ‖ π(<i>K</i>, <i>R</i><sub>i</sub><i>X</i><sub>j</sub> + <i>t</i><sub>i</sub>) − <i>x</i><sub>ij</sub> ‖<sup>2</sup> )", 2)
B("with a Huber loss and the sparse Jacobian structure (each residual depends on one camera and one point), using SciPy’s "
  "trust-region least squares [7]. After each registration it lowers the RMS error by roughly 5–35%.")
H2("Point cloud, visualisation and reprojection")
B("Point colours are sampled from the images; clouds are exported as PLY and inspected with Matplotlib, Open3D and a self-contained "
  "interactive WebGL page. For every camera the 3D points are projected back,")
EQ("<i>x</i><sub>i</sub> ~ <i>P</i><sub>i</sub> <i>X</i>,     <i>P</i><sub>i</sub> = <i>K</i>[<i>R</i><sub>i</sub> | <i>t</i><sub>i</sub>]", 3)
B("and compared with the detected features (Fig. 4). A large reprojection error indicates a wrong camera pose, a poorly triangulated "
  "point (small parallax), a false match or a wrong calibration. Points seen in only two views satisfy the constraints almost by "
  "construction, so we also report the error for points seen in three or more views.")
FIG("fig4_reprojection.png", "Reprojection check. (a) Detected features (green circles) and reprojected 3D points (red); the offset "
    "between them is magnified ×15 so it is visible. (b) Error histogram over all observations of the baseline (1600 px images).")
H2("Evaluation protocol")
B("We align our model to COLMAP by a Sim(3) transform (Umeyama [12], inside RANSAC, with an optical-axis term when camera centres are "
  "nearly collinear) and report the camera position error relative to the trajectory extent, the rotation error, the fraction of our "
  "points within 2% of that extent from a COLMAP point, and completeness. Runtimes are for a laptop with an RTX 3050 6 GB GPU.")
H2("Dense stereo reconstruction (advanced)")
B("Given the poses, neighbouring registered cameras are rectified so epipolar lines become image rows, matched by semi-global "
  "matching [9] (block 5) within a disparity range derived from the sparse depths, and back-projected. Depth maps are fused by "
  "voxel voting: a voxel is kept only if at least two different stereo pairs place a point in it.")
H2("Learned features and GPU matching (advanced)")
B("DISK [10] detects keypoints with 128-D descriptors and LightGlue [11] matches them with a transformer that reasons over both "
  "images; both are pretrained (no training by us) and run through Kornia [14] on the GPU. We also ported our brute-force "
  "ratio-plus-mutual matcher to the GPU; it returns identical matches and is 10.9× faster per pair (0.029 s versus 0.319 s). "
  "Because learned matching gives true pairs thousands of inliers, we raise the pair-acceptance threshold from 25–40 to "
  "150 inliers when using it (Section V-G).")
H2("Self-calibration of the phone camera (own captures)")
B("The phone is uncalibrated. EXIF gives the 35 mm-equivalent focal length (24 mm), which converts to pixels through the image "
  "diagonal, <i>f</i> = 24 · diag / 43.27 (the common long-side/36 mm shortcut is right only for 3:2 images and was 4% too "
  "small here). We refine it by self-calibration from fundamental matrices: for the true <i>K</i>, <i>E</i> = <i>K</i><sup>T</sup> "
  "<i>F K</i> has two equal singular values, so for a grid of <i>f</i> (square pixels, principal point at the image centre) we compute")
EQ("<i>c</i>(<i>f</i>) = (<i>σ</i><sub>1</sub> − <i>σ</i><sub>2</sub>) / (<i>σ</i><sub>1</sub> + <i>σ</i><sub>2</sub>),     "
   "<i>σ</i> = svd(<i>K</i>(<i>f</i>)<sup>T</sup> <i>F K</i>(<i>f</i>))", 4)
B("over 40–60 RANSAC-verified pairs and take the <i>f</i> minimising the median of <i>c</i>. On south-building this recovers COLMAP’s "
  "calibrated focal length to 0.5% (1326.5 against 1333.2 px from 38 pairs). On the four own sets it gives 1137–1159 px at "
  "1600 px, 2.5–4.5% above the EXIF value and consistent across sets; all own reconstructions use a common factor of 1.04. "
  "An independent chessboard calibration of the same phone (40 photos, 25 mm squares) gives 1156 px, within 0.2% of this value "
  "(1153–1163 px across distortion models and photo subsets); its fit error is high (RMS 6 px) because the oblique views were "
  "defocused and the paper was not perfectly flat, so we quote the agreement as about ±1%. "
  "Two alternatives failed: choosing <i>f</i> by reprojection error was inconclusive (fewer images register at low <i>f</i>, which "
  "lowers the error spuriously), and refining <i>f</i> inside bundle adjustment stayed at its wrong starting value (±12%) on "
  "south-building, probably because focal length and depth are weakly separable for a near-planar scene.")

# ======================================================================================== IV. EXPERIMENTAL DESIGN
H1("Experimental Design")
B("All experiments share one control: 12 south-building images (P1180181–P1180203, stride 2), RootSIFT with 8000 features, "
  "1600 px, ratio 0.8 plus mutual check, RANSAC at 1 px, PnP and bundle adjustment. Each changes exactly one variable. Metrics: "
  "images registered, 3D points (and points seen in at least three views), mean track length, reprojection error, camera position "
  "and rotation error, completeness, verified inliers per pair and runtime.")
BUL("<b>A – number of images</b> (N = 3–20, nested subsets). Hypothesis: more points with diminishing returns; runtime "
    "roughly quadratic from all-pairs matching.")
BUL("<b>B – detector</b> (SIFT, RootSIFT, ORB, AKAZE, BRISK). Hypothesis: SIFT-type descriptors beat binary ones on wide "
    "baselines.")
BUL("<b>C – viewpoint change</b> (frame stride 1–6 with 8 images, plus a two-view study). Hypothesis: stride 1 gives poor "
    "depth, very large strides break matching, with an optimum between.")
BUL("<b>D – image quality</b> (blur σ = 1–8 px, noise σ = 10–50, resolution 1200–200 px, contrast "
    "×0.5–0.1, gamma 2.0–3.5). Hypothesis: blur and low resolution hurt most; contrast changes little because "
    "descriptors are gradient-normalised.")
B("Challenging scenarios use 11 Person Hall cases with an outcome rule fixed in advance (success: at least 90% of images registered "
  "and position error below 2%) and five own phone captures, one of them a deliberate negative case. The advanced work compares "
  "classical, GPU and learned matching and COLMAP.")

# ======================================================================================== V. RESULTS
H1("Results")
H2("Baseline reconstruction")
B("All twelve images register (Table I). Fig. 3 shows the cloud with camera positions and orientations; Fig. 4 the reprojection "
  "check: mean error 0.14 px, median 0.10 px at 1600 px (about 0.27 px at the original resolution), 0.16 px for points seen in at "
  "least three views. Poses agree with COLMAP to 0.11% (position) and 0.18° (rotation).")
TABLE("Baseline reconstruction (12 images)", ["Quantity", "Value"],
      [["Images registered", "12 / 12"], ["3D points (seen in ≥ 3 views)", "10,934 (4,986)"], ["Mean track length", "2.95"],
       ["Reprojection error, all obs. (mean / median)", "0.138 / 0.103 px"], ["Reprojection error, ≥ 3-view (mean)", "0.158 px"],
       ["Camera position error (rel. to extent)", "0.11 %"], ["Camera rotation error (mean / max)", "0.18° / 0.37°"],
       ["Points within 2% of a COLMAP point", "99.4 %"], ["Run time", "about 1 min"]], [3.9, 2.0])
H2("Experiment A: number of images")
B("Points grow from 1.4 k to 23 k and runtime roughly quadratically (212 s at N = 20), as hypothesised (Table II, Fig. 5a). "
  "<b>Accuracy is not monotonic</b>: position error is best at N = 8 (0.08%) and worse at N = 16 and 20 (0.41%, 0.29%); weaker-overlap "
  "frames and accumulated drift are plausible causes but were not isolated. The mean reprojection error rises from 0.11 to 0.18 px "
  "as tracks lengthen: with N = 3 only 21 points are seen in three views and two-view points fit almost exactly. N = 4 registers "
  "only three images (a link of 9–10 inliers is noise) and its accuracy score is unreliable because three nearly collinear "
  "cameras make the Sim(3) alignment ill-posed; a camera-independent ICP alignment places 78–80% of its points within 2%, so it "
  "is a fragile, not a wrong, reconstruction.")
TABLE("Experiment A: number of images", ["N", "Reg.", "Points", "≥ 3 views", "Cam. err %", "Time s"],
      [["3", "3/3", "1,448", "21", "0.17", "1.5"], ["4", "3/4", "2,232", "70", "0.62<sup>a</sup>", "3.7"], ["6", "6/6", "6,589", "2,237", "0.13", "14"],
       ["8", "8/8", "8,889", "4,097", "0.08", "28"], ["12", "12/12", "10,934", "4,986", "0.11", "62"],
       ["16", "16/16", "18,849", "9,193", "0.41", "139"], ["20", "20/20", "23,294", "12,375", "0.29", "212"]],
      [0.7, 1.0, 1.3, 1.3, 1.3, 1.0], note="<sup>a</sup>Alignment on three nearly collinear cameras is unreliable.", left_cols=())
H2("Experiment B: feature detector")
B("RootSIFT is best (Table III): it registers all images and yields 483 verified inliers per pair; plain SIFT gives 452 and 11/12. "
  "ORB, the fastest, finds four times fewer inliers, registers 8/12, has a reprojection error of 0.35 px and about ten times the camera "
  "error; AKAZE and BRISK lie between. The hypothesis holds, with the caveat that the binary detectors were not tuned.")
TABLE("Experiment B: detectors (same 12 images)", ["Detector", "Reg.", "Points", "Inliers/pair", "Reproj. px", "Cam. err %"],
      [["SIFT", "11/12", "9,414", "452", "0.129", "0.12"], ["<b>RootSIFT</b>", "<b>12/12</b>", "<b>10,934</b>", "<b>483</b>", "0.138", "0.11"],
       ["ORB", "8/12", "3,706", "115", "0.351", "1.23"], ["AKAZE", "8/12", "5,112", "201", "0.222", "0.20"],
       ["BRISK", "9/12", "6,642", "271", "0.209", "0.13"]], [1.5, 1.0, 1.3, 1.4, 1.3, 1.3])
H2("Experiment C: viewpoint change")
B("All eight images register at strides 1–2, six at stride 3 and only 2–3 at strides 4–6, while verified inliers fall "
  "from 988 to 140 (Fig. 5b). A two-view study (frames <i>i</i>, <i>i</i>+<i>k</i>) gives 2,520, 1,270 and 377 inliers at median "
  "parallax 12.6°, 24° and 34°, and noise level (9–10) beyond: <b>matching collapses near 35° of viewpoint "
  "change</b>. Accuracy is lowest at stride 1 (92% of points within 2% against 99%), consistent with weak depth from small "
  "baselines. Even consecutive south-building frames have about 12° parallax, so the true small-baseline regime is not reached.")
FIG("fig5_expA_expC.png", "(a) Experiment A: 3D points (blue) and camera position error (red) against the number of images. "
    "(b) Experiment C: images registered (bars, of 8) and verified inliers per pair (line) against frame stride.")
H2("Experiment D: image quality")
B("Degradation appears as lost points and images rather than pose drift: for runs that still register, camera error stays at "
  "0.1–0.4% (Fig. 6). Blur is the most destructive: σ = 2 halves the points, σ = 4 leaves 7% and 8/12 images. Noise "
  "σ = 25 leaves 14% (8/12) and σ = 50 fails. Down to 800 px keeps 47% of the points; 400 px keeps 8%. Contrast ×0.25 "
  "still registers 11/12 but ×0.1 cannot initialise; darkening by gamma does not hurt. The hypothesis holds.")
FIG("fig6_expD.png", "Experiment D: 3D points kept (% of the undegraded baseline) for each degradation; numbers show images "
    "registered of 12.")
H2("Dense reconstruction")
B("Fifteen rectified stereo pairs from 16 consecutive images give 428,018 fused points, 23.5× the sparse cloud (18,216), fused in "
  "18 s (Fig. 7). In the reconstructed volume 66.4% of COLMAP’s points have one of ours within 2% of the scene radius (sparse "
  "58.5%) and the median gap to COLMAP’s surface points falls from 1.4% to 0.5%; individual dense points are about 35% noisier than "
  "sparse ones (per-pixel disparity versus sub-pixel features). Fusion support is the main quality knob: requiring one pair gives "
  "1.1 M points but only 46% on the dominant facade plane; three pairs give 224 k points with 80%. Pairs three frames apart "
  "collapse (76 points), matching the limit of Experiment C.")
FIG("fig7_dense.png", "Sparse (a) and dense (b) clouds of the same 16-image sequence, front view.")
H2("Learned features and comparison with COLMAP")
B("On image pairs scored with COLMAP’s geometry (Fig. 8a), correct inliers under sun flare rise from 379 (RootSIFT) to 3,056 "
  "(DISK with the ratio test) and 3,400 (DISK + LightGlue); for foliage the medians are 4, 28 and 229, and for scale change 174, 809 "
  "and 1,124. Of 13 weak links that cut the whole building loop, 0, 5 and 8 are bridged (at least 25 correct matches). Wide "
  "viewpoint change still defeats every method. In full reconstructions (Table IV) learned matching yields 1.8× the points and "
  "4.6× the inliers per pair but a 2.7× higher reprojection error and about twice the camera error, as DISK keypoints lie on the "
  "integer pixel grid.")
B("<b>A second silent failure.</b> At strides 4–6 learned matching registered more images (6, 4, 8 of 8) but the models were wrong "
  "(59–85% camera error). In the stride-6 case 8 of 19 verified pairs were false, with a median of only 68 inliers against 2,357 for correct pairs. "
  "Our SIFT-era thresholds could not separate them; raising the pair threshold to 150 inliers removed them (3–4 images "
  "registered, nearly all correct).")
B("Compared with COLMAP on identical images (Table V; its model restricted to points seen by our images), poses agree to 0.03–0.11%. "
  "Our point counts are 0.6–2.3× COLMAP’s with similar track lengths. Our lower reprojection error is <i>not</i> evidence of "
  "superiority: our intrinsics are COLMAP’s refined calibration, its error is averaged over a model constrained by all 128 images, "
  "and our acceptance thresholds differ. COLMAP additionally uses retrieval-based matching, two-view geometry classification, "
  "a more elaborate next-view selection, local bundle adjustment, intrinsics refinement, re-triangulation, sub-model merging and a "
  "PatchMatch dense stage; our pipeline has none of these (it registers the image with the most correspondences and densifies "
  "with SGBM).")
TABLE("Classical, GPU and learned matching (south-building, 12 images)", ["Quantity", "RootSIFT CPU", "RootSIFT GPU", "DISK + LightGlue"],
      [["3D points", "10,934", "10,938", "<b>20,084</b>"], ["Points ≥ 3 views", "4,986", "4,989", "<b>12,501</b>"],
       ["Inliers per pair", "483", "483", "<b>2,245</b>"], ["Mean track length", "2.95", "2.95", "3.83"],
       ["Reprojection (px)", "<b>0.138</b>", "0.138", "0.379"], ["Camera error (%)", "<b>0.11</b>", "0.12", "0.20"],
       ["Run time (s)", "78", "68", "116"]], [2.1, 1.4, 1.4, 1.9])
TABLE("Ours versus COLMAP on identical images", ["Images", "Pts ours", "Pts COLMAP", "Track o/C", "Reproj. o/C (px)", "Cam. err %"],
      [["south 12", "10,934", "8,830", "2.95/2.69", "0.27/0.48", "0.11"], ["south 12, DISK", "20,084", "8,830", "3.83/2.69", "0.73/0.48", "0.20"],
       ["south 16", "18,216", "11,577", "3.89/3.82", "0.29/0.46", "0.05"], ["hall A", "12,584", "17,780", "3.03/3.07", "0.39/0.60", "0.03"],
       ["hall C", "6,509", "10,777", "3.33/4.61", "0.52/0.78", "0.11"]], [1.5, 1.1, 1.3, 1.3, 1.5, 1.0],
      note="Reprojection errors at original resolution; ours uses COLMAP’s intrinsics.")
FIG("fig9_learned_colmap.png", "(a) Median correct RANSAC inliers per pair for three matchers on hard pairs (log scale). "
    "(b) Plan view of our 12-image cloud (red) and COLMAP’s points seen by the same images (blue), Sim(3)-aligned.")

# ======================================================================================== VI. CHALLENGING SCENARIOS
H1("Challenging Scenarios and Failure Analysis")
B("Person Hall provides foliage, repetition, sun flare and a closed loop. We ran the unchanged default pipeline on 11 cases "
  "(Table VI) and logged every outcome.")
TABLE("Person Hall cases", ["Case", "Tests", "Reg.", "Cam. err", "Outcome"],
      [["A facade", "reference", "12/12", "0.03%", "success"], ["B foliage", "occlusion", "2/12", "n/a", "<b>failure</b>"],
       ["C arches+flare", "repetition", "12/12", "0.11%", "success†"], ["D corner turn", "scale change", "11/12", "0.07%", "success"],
       ["Stride 2–8", "viewpoint", "8/8 (×4)", "≤ 0.75%", "success"], ["F loop, 55", "whole loop", "15/55", "0.66%", "<b>failure</b>"],
       ["F3 loop, 110", "whole loop", "44/110", "0.16%", "<b>failure</b> (partial)"],
       ["F3, learned", "whole loop", "59/110", "53%", "<b>failure</b> (drift)"]],
      [1.8, 1.4, 1.3, 1.1, 2.1], note="†After fixing the EXIF bug; the first run of C was 12/12 registered but 51% wrong.", left_cols=(0, 1))
B("<b>Foliage (case B).</b> Only 2 of 12 images register (63 points). Fig. 9a: a tree hides the wall; the 74 surviving matches are "
  "all correct against COLMAP geometry but too few and confined to the visible wall—a failure by occlusion, not by false matches.")
B("<b>A silent failure caused by our loader (case C).</b> The first run registered all 12 images with 0.6 px reprojection error and "
  "6.5 k points, yet its camera error was 51% and its rotation error 139°. 50 of 60 verified pairs contradicted COLMAP although "
  "their matches looked right, and COLMAP’s relative rotation between the consecutive sampled frames IMG_1228 and IMG_1230 was 91°. "
  "48 of the 330 photographs carry an EXIF portrait tag: OpenCV’s imread silently rotates them, while COLMAP (and our single "
  "landscape camera matrix) uses the stored pixels. With IMREAD_IGNORE_ORIENTATION the error fell to 0.11% (case D from 0.97% to "
  "0.07%). South-building has no tagged photographs, so earlier results are unaffected. A low reprojection error and full "
  "registration therefore do not validate a reconstruction.")
B("<b>Corner turn and viewpoint sweep.</b> In case D, IMG_1313 is lost; the photographer stepped back, so it shows the corner from about "
  "twice the distance (a likely scale-change failure, not measured). All four stride sweeps succeed and accuracy slips only at stride "
  "8 (0.75%): stride is not comparable across datasets, parallax is.")
B("<b>Whole loop.</b> With every sixth image (55) only 15 register: the pair graph (edge = at least 25 verified inliers) has 15 connected "
  "components (Fig. 9b), so incremental SfM cannot cross the gaps. Every third image (110) registers 44 cameras, accurately (0.16%) "
  "but covering one side; relaxed thresholds merge the graph into 3 components yet registration stays at 45 of 110: connectivity is "
  "necessary but not sufficient, since a new image also needs 12–20 matches with existing 3D points. Learned matching "
  "registers 59 cameras but only 10 are correct; errors grow smoothly along the chain (Fig. 9c) although 372 of 403 verified pairs "
  "are individually correct (median rotation error 0.62°): accumulated drift without loop closure, not false links. The cause "
  "(for example coarse keypoint localisation) was not isolated by a controlled test.")
FIG("fig8_failures.png", "Failure evidence. (a) Foliage pair: all 74 surviving matches are correct but too few. (b) Whole loop "
    "F: verified inliers per image pair (log scale) fall into disconnected clusters. (c) Learned loop F3: rotation error of each "
    "registered camera along the chain (green = correct, red = wrong).")
B("<b>Own captures.</b> All 330 photographs register (Table VII) and independent geometry checks agree: the camera centres of "
  "each orbit form a near-circular path with 369–392° of angular coverage, 5–9% radius spread and 1–3% height "
  "spread, and the two yellow-sign rings sit consistently in one model. All-pairs matching also links the start and end of each orbit, "
  "which windowed matching would not. With no ground truth, quality is judged by reprojection error (0.38–0.59 px at 1600 px, "
  "against 0.14 px on south-building with COLMAP’s calibration, i.e. the residual limits of an uncalibrated phone), track "
  "statistics and inspection. The yellow sign, fire extinguisher and snacks are positive examples (Fig. 10): both faces of the sign "
  "with legible text, the extinguisher’s label, valve and handle, and the wrinkled snack bag.")
TABLE("Own captures (iPhone 16 Pro Max)", ["Set", "Imgs", "Reg.", "Points", "Track", "Reproj. px", "Dense pts"],
      [["yellow sign", "119", "119", "28,131", "3.91", "0.59", "322,793"], ["fire extinguisher", "83", "83", "22,995", "4.63", "0.43", "576,819"],
       ["snacks", "71", "71", "44,621", "4.18", "0.42", "1,033,713"], ["bottle, classical", "57", "57", "14,254", "3.38", "0.38", "35,330"],
       ["bottle, DISK+LG", "57", "57", "67,129", "2.98", "0.57", "25,188"]], [2.0, 0.9, 0.8, 1.2, 1.0, 1.2, 1.4],
      note="Dense points: inside a cylinder around the object.")
FIG("fig10_own_positive.png", "Own captures, positive examples: input photo (left) and dense reconstruction seen from the same direction "
    "(right) for (a) the yellow sign (both heights), (b) the fire extinguisher and (c) the snacks.")
B("<b>Dense-stage lessons.</b> The first dense result was a diffuse blob. Two causes were found by looking at it: the fusion voxel was "
  "sized from the whole office, far too coarse for a small object, and neighbouring frames are only about 5° apart on these orbits, "
  "too small a stereo baseline. Fusing only a cylinder around the orbit axis with a voxel sized to it, and pairing frames two to three "
  "apart (10–16°), fixed both. The glossy, uniformly red extinguisher also needed more permissive stereo settings "
  "(13,700 to 576,819 points).")
B("<b>Negative case: transparent bottle.</b> It registers 57/57 with both pipelines, yet geometry exists only where the object is "
  "opaque: the red label and the black cap reconstruct densely, while the clear body is almost empty or noise (Fig. 11), because the "
  "camera sees the background through it and matching assumes one appearance per surface. DISK + LightGlue gives the same result with "
  "4.7× more sparse points but a higher reprojection error (0.57 against 0.38 px) and four times the run time: learned features "
  "do not rescue transparency.")
FIG("fig11_own_negative.png", "Negative case, transparent bottle: (a) input photo; dense reconstruction with (b) classical and "
    "(c) DISK + LightGlue matching. Only the opaque label and cap are recovered.")

# ======================================================================================== VII. DISCUSSION
H1("Discussion")
B("<b>Viewpoint and features.</b> Parallax, not frame stride, controls quality: too little and depth is poorly conditioned (stride 1 "
  "accuracy 92% against 99%), too much (about 35°) and matching fails. Learned features move this limit slightly (stride "
  "3–4 on south-building) but not fundamentally, and they trade localisation precision for match count. Scene properties "
  "(foliage, repetition, loops) caused every failure; none was fixed by a threshold.")
B("<b>Reconstruction errors.</b> Reprojection error is necessary but not sufficient: two-view points fit by construction and a "
  "globally wrong model can have a small error. Registration count is not quality either (SIFT loop: 44 accurate cameras; learned "
  "loop: 59 cameras, 49 wrong). Independent validation found both the EXIF bug and the wrong learned models.")
B("<b>Missing regions and limitations.</b> Foliage, sky and glass produce no points; dense stereo is unreliable on thin structures; COLMAP "
  "covers ground and foliage that we miss. Limitations: one main sequence; RANSAC variation not averaged (re-runs reproduced "
  "identical numbers); COLMAP is a reference, not truth; intrinsics are not estimated; no loop closure, local bundle adjustment or "
  "sub-model merging; binary detectors untuned; mainly building-scale data.")
B("<b>Own captures.</b> Careful capture mattered more than algorithm choice: one lens, constant distance and overlapping neighbours "
  "(hundreds of verified matches) let all five sets reconstruct without per-set tuning, unlike the Person Hall loop. The limits were "
  "physical: transparent and glossy surfaces break the assumptions of matching and stereo, and an uncalibrated phone caps accuracy. "
  "Without ground truth we cannot state a metric accuracy for these sets; a chessboard calibration of the same lens and focus "
  "distance would provide the missing reference.")
B("<b>Improvements.</b> Loop-closure detection by image retrieval; local bundle adjustment and merging of sub-models; sub-pixel "
  "refinement of learned keypoints; vegetation masking; denser sampling; and GPU matching of all pairs (about 6.5 h on CPU for the full "
  "loop, an estimated 26 min at the measured GPU speed).")

# ======================================================================================== VIII. CONCLUSION
H1("Conclusion")
B("An incremental SfM pipeline implemented from the geometry up reproduces COLMAP’s camera poses to about 0.1% when matches are "
  "good, and a rectified-SGBM stage densifies the result 23-fold. Controlled experiments show that more images do not always help, "
  "that RootSIFT is the best detector tested, that matching collapses near 35° of viewpoint change and that blur is the most "
  "damaging degradation. The most important lessons concern evaluation: reprojection error and registration counts can hide wrong "
  "reconstructions, as an orientation bug and learned-matching drift showed. Learned features are valuable on hard pairs but not a "
  "free upgrade: they need stricter verification and gave noisier geometry. Our own phone captures reconstruct completely after "
  "self-calibration, except that a transparent bottle yields only its opaque parts.")

# ======================================================================================== AI DECLARATION + REFERENCES
new_p("AI Usage Declaration", "Heading 5")
B("<hl>[YOU: review every statement and edit it so it is true for you.]</hl> <b>Tools:</b> Claude (Claude Code, model Claude Sonnet 5.5, "
  "Anthropic). <b>Used for:</b> writing and debugging the pipeline, experiment and analysis scripts; producing figures, videos and the "
  "interactive viewer; diagnosing failures (for example the EXIF bug); and drafting the progress log, the demo script and this report. "
  "<b>AI-assisted parts:</b> most code in the repository folders sfm/, scripts/ and experiments/, and first drafts of the text. "
  "<b>Verification and modification:</b> all numbers were produced by running the code; reconstructions were checked against the "
  "COLMAP ground truth; pose recovery and triangulation were cross-checked against OpenCV; failures were investigated instead of being "
  "reported at face value. <hl>[YOU: state what you personally read, ran, changed and can explain, and any other AI tool used.]</hl>")
new_p("References", "Heading 5")
REF("R. Hartley and A. Zisserman, <i>Multiple View Geometry in Computer Vision</i>, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2004.")
REF("D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” <i>Int. J. Comput. Vis.</i>, vol. 60, no. 2, pp. 91–110, 2004.")
REF("R. Arandjelović and A. Zisserman, “Three things everyone should know to improve object retrieval,” in <i>Proc. IEEE CVPR</i>, 2012, pp. 2911–2918.")
REF("D. Nistér, “An efficient solution to the five-point relative pose problem,” <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 26, no. 6, pp. 756–770, 2004.")
REF("M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography,” <i>Commun. ACM</i>, vol. 24, no. 6, pp. 381–395, 1981.")
REF("V. Lepetit, F. Moreno-Noguer, and P. Fua, “EPnP: An accurate O(n) solution to the PnP problem,” <i>Int. J. Comput. Vis.</i>, vol. 81, no. 2, pp. 155–166, 2009.")
REF("B. Triggs, P. McLauchlan, R. Hartley, and A. Fitzgibbon, “Bundle adjustment—A modern synthesis,” in <i>Vision Algorithms: Theory and Practice</i>, LNCS 1883. Berlin, Germany: Springer, 2000, pp. 298–372.")
REF("J. L. Schönberger and J.-M. Frahm, “Structure-from-motion revisited,” in <i>Proc. IEEE CVPR</i>, 2016, pp. 4104–4113.")
REF("H. Hirschmüller, “Stereo processing by semiglobal matching and mutual information,” <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 30, no. 2, pp. 328–341, 2008.")
REF("M. Tyszkiewicz, P. Fua, and E. Trulls, “DISK: Learning local features with policy gradient,” in <i>Proc. NeurIPS</i>, 2020.")
REF("P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “LightGlue: Local feature matching at light speed,” in <i>Proc. IEEE/CVF ICCV</i>, 2023.")
REF("S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 13, no. 4, pp. 376–380, 1991.")
REF("G. Bradski, “The OpenCV library,” <i>Dr. Dobb’s J. Softw. Tools</i>, 2000.")
REF("E. Riba, D. Mishkin, D. Ponsa, E. Rublee, and G. Bradski, “Kornia: An open source differentiable computer vision library for PyTorch,” in <i>Proc. IEEE WACV</i>, 2020.")
REF("J. L. Schönberger, “COLMAP datasets (South Building, Person Hall).” [Online]. Available: https://demuc.de/colmap/datasets/")

# remove leftover template instruction text boxes (e.g. "We suggest that you use a text box to insert a graphic ...").
# The text-box content element sits in a different namespace prefix after conversion, so match on the local tag name.
for _r in list(doc.element.body.iter(qn("w:r"))):
    if any(isinstance(e.tag, str) and e.tag.endswith("}txbxContent") for e in _r.iter()):
        _tx = "".join(t.text or "" for t in _r.iter(qn("w:t"))).lower()
        if "text box" in _tx or "funding agency" in _tx:
            _r.getparent().remove(_r)
for _s in doc.sections:
    for _ft in (_s.footer, _s.first_page_footer, _s.even_page_footer):
        for _fp in _ft.paragraphs:
            if "XXX-X" in _fp.text:
                for _r in _fp.runs:
                    _r.text = ""
doc.core_properties.title = "Multi-View 3D Reconstruction and Point Cloud Generation (draft)"
doc.core_properties.author = ""
doc.save(OUT)
print("saved", OUT)
