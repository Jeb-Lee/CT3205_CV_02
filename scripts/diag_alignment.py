import sys, numpy as np
sys.path.insert(0, ".")
from sfm import datasets as D, geometry as G, evaluate as Ev
from sfm.reconstruction import Reconstruction, SfMConfig
ds = D.load_dataset("south-building")
names = D.select_images(ds, 40, 4, 2); print(names)
rec = Reconstruction(ds, names, SfMConfig(), cache_dir="results/cache", verbose=False).run()
print("registered", sorted(rec.cams), "init", rec.init_pair)
for m in rec.log["events"]: print({k: (round(v,3) if isinstance(v,float) else v) for k,v in m.items()})
ids = sorted(rec.cams)
Ce = np.array([G.camera_center(*rec.cams[i]) for i in ids]); Cg = np.array([G.camera_center(*ds.gt_poses[names[i]]) for i in ids])
print("sv ratio gt", np.linalg.svd(Cg-Cg.mean(0), compute_uv=False)); print("sv ratio est", np.linalg.svd(Ce-Ce.mean(0), compute_uv=False))
# is the unregistered image linked?
for (i,j),m in sorted(rec.matches.items()): print((i,j), m["n_inl"])
# force axis-augmented alignment
d = np.linalg.norm(Ce-Ce.mean(0),axis=1).max()
s,R,t = Ev.umeyama(Ce,Cg)
axe = np.array([rec.cams[i][0].T@[0,0,1.] for i in ids]); axg = np.array([ds.gt_poses[names[i]][0].T@[0,0,1.] for i in ids])
s2,R2,t2 = Ev.umeyama(np.vstack([Ce,Ce+d*axe]), np.vstack([Cg,Cg+s*d*axg]))
pc = rec.point_cloud(with_colors=False)
from scipy.spatial import cKDTree
tree = cKDTree(ds.gt_points); ext = np.linalg.norm(Cg-Cg.mean(0),axis=1).max()
for lab,(ss,RR,tt) in {"centres only":(s,R,t),"centres+axes":(s2,R2,t2)}.items():
    dd,_ = tree.query(ss*pc["xyz"]@RR.T+tt); print(lab, "scale",round(ss,3),"within2%",np.mean(dd<0.02*ext).round(3),"median",np.median(dd)/ext)

print("--- scaled ICP (camera-independent) ---")
def icp(P, Q, tree, s, R, t, iters=40, trim=0.6):
    for _ in range(iters):
        X = s*P@R.T+t
        d, idx = tree.query(X)
        keep = d <= np.quantile(d, trim)
        s, R, t = Ev.umeyama(P[keep], Q[idx[keep]])
    return s, R, t
for lab,(ss,RR,tt) in {"from centres":(s,R,t),"from centres+axes":(s2,R2,t2)}.items():
    s3,R3,t3 = icp(pc["xyz"], ds.gt_points, tree, ss, RR, tt)
    dd,_ = tree.query(s3*pc["xyz"]@R3.T+t3)
    print(lab, "-> ICP scale", round(s3,3), "within2%", np.mean(dd<0.02*ext).round(3), "median/extent", (np.median(dd)/ext).round(4))
    # implied camera error after ICP
    Ca = s3*Ce@R3.T+t3; print("   cam pos rmse/extent", (np.sqrt(((Ca-Cg)**2).sum(1).mean())/ext).round(4))
