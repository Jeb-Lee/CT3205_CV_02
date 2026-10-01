"""Re-render turntable videos + scene figures for Person Hall cases WITHOUT touching cases.csv or the recorded stats.

    python scripts/rerender_videos.py A F3 F3_dls
(Needed after fixing viz.level_matrix: the first renders orbited the cameras' viewing axis instead of the vertical.)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))
from sfm import datasets as D, viz
from sfm.reconstruction import Reconstruction, SfMConfig
from run_person_hall import CASES, OUT, CACHE

ds = D.load_dataset("person-hall")
for key in sys.argv[1:]:
    title, why, start, count, stride, over = CASES[key]
    names = D.select_images(ds, start, count, stride)
    rec = Reconstruction(ds, names, SfMConfig(**over), cache_dir=CACHE, verbose=False).run()
    pc = rec.point_cloud()
    out = os.path.join(OUT, key)
    viz.plot_scene(rec, pc, os.path.join(out, "scene.png"), views=((25, -60), (90, -90)),
                   title=f"case {key} ({title}): {len(pc['xyz'])} pts, {len(rec.cams)}/{len(names)} cameras")
    if len(rec.cams) >= 6:
        viz.turntable_video(rec, pc, os.path.join(out, "turntable.mp4"), frames=150, size=(1280, 720), fps=30, elev=12, point_size=2.5)
    print("rerendered", key, len(rec.cams), "cameras", flush=True)
