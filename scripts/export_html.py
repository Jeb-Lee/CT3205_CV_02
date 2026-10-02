"""Export a reconstruction as ONE self-contained interactive HTML file (raw WebGL, no libraries, works offline).

    python scripts/export_html.py --folder results/dense --rec results/cache/rec_south-building_40_16_1.pkl \
        --out results/viewer/south_building.html

Contains: dense cloud (subsampled), sparse cloud, camera frusta. Controls in the page: drag = rotate,
right-drag / shift-drag = pan, wheel / pinch = zoom, checkboxes, point-size slider, view presets, auto-rotate.
"""
import argparse
import base64
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sfm import viz, geometry as G  # noqa: E402

def level_rotation(rec):
    """Rotation taking the SfM world (first-camera frame, arbitrary tilt) to a viewer frame with y up.

    Photographers hold cameras roughly upright, so the mean of the cameras' up axes (-y in camera coords) is 'up';
    the mean viewing direction (projected to the horizontal) becomes -z, so the default eye sits behind the cameras.
    """
    ups = np.array([-(R.T @ np.array([0, 1.0, 0])) for R, t in rec.cams.values()])
    fwd = np.array([R.T @ np.array([0, 0, 1.0]) for R, t in rec.cams.values()])
    Y = ups.mean(0); Y /= np.linalg.norm(Y)
    f = fwd.mean(0); f -= (f @ Y) * Y; f /= np.linalg.norm(f)
    Z = -f
    X = np.cross(Y, Z)
    return np.vstack([X, Y, Z])          # rows = viewer axes expressed in world coords (proper rotation)


def read_ply(path):
    raw = open(path, "rb").read()
    k = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(raw[:k].split(b"element vertex ")[1].split()[0])
    rec = np.frombuffer(raw[k:k + 15 * n], dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    return rec["p"].astype(np.float64), rec["c"].copy()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f4f2;--fg:#1c1c1c;--panel:rgba(255,255,255,.92);--line:#d6d6d0;--acc:#b23a2f}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--panel:rgba(30,30,30,.92);--line:#3a3a3a}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
canvas{position:fixed;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}
#ui{position:fixed;top:12px;left:12px;width:230px;max-width:calc(100vw - 24px);background:var(--panel);border:1px solid var(--line);
 border-radius:10px;padding:12px 14px;backdrop-filter:blur(6px)}
#ui h1{font-size:15px;margin:0 0 2px}#ui p{margin:0 0 8px;color:#777;font-size:12px}
#ui label{display:flex;align-items:center;gap:8px;margin:5px 0;cursor:pointer}
#ui .row{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 2px}
#ui button{font:inherit;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;padding:4px 9px;cursor:pointer}
#ui button:hover{border-color:var(--acc)}
#ui input[type=range]{width:100%}
#stats{margin-top:8px;font-size:12px;color:#777}
#hint{position:fixed;bottom:10px;left:0;right:0;text-align:center;font-size:12px;color:#888;pointer-events:none}
</style></head><body>
<canvas id="c"></canvas>
<div id="ui">
 <h1>__TITLE__</h1><p>__SUB__</p>
 <label><input type="checkbox" id="cDense" checked> Dense cloud</label>
 <label><input type="checkbox" id="cSparse"> Sparse SfM cloud</label>
 <label><input type="checkbox" id="cCams" checked> Cameras (frusta)</label>
 <label>Point size <input type="range" id="ps" min="1" max="8" step="0.5" value="2"></label>
 <div class="row"><button data-v="front">Front</button><button data-v="corner">Corner</button>
 <button data-v="side">Side</button><button data-v="top">Top</button></div>
 <div class="row"><button id="spin">Auto-rotate</button><button id="bg">Background</button></div>
 <div id="stats"></div>
</div>
<div id="hint">drag: rotate &nbsp;·&nbsp; right-drag / shift-drag: pan &nbsp;·&nbsp; wheel / pinch: zoom</div>
<script>
const DATA=__DATA__;
function dec(b,T){const s=atob(b),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return T?new T(u.buffer):u}
const cv=document.getElementById('c'),gl=cv.getContext('webgl',{antialias:true});
if(!gl)document.body.insertAdjacentHTML('beforeend','<p style="padding:80px">WebGL is not available in this browser.</p>');
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o}
function prog(v,f){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,v));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);return p}
const P1=prog('attribute vec3 a;attribute vec3 c;uniform mat4 m;uniform float s;varying vec3 v;void main(){gl_Position=m*vec4(a,1.);gl_PointSize=s;v=c;}',
 'precision mediump float;varying vec3 v;void main(){gl_FragColor=vec4(v,1.);}');
const P2=prog('attribute vec3 a;uniform mat4 m;void main(){gl_Position=m*vec4(a,1.);}',
 'precision mediump float;uniform vec3 k;void main(){gl_FragColor=vec4(k,1.);}');
function buf(arr){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);return b}
const L={};
for(const k of ['dense','sparse']){const d=DATA[k];L[k]={n:d.n,pos:buf(dec(d.pos,Float32Array)),col:buf(dec(d.col))}}
const camPos=dec(DATA.cams.pos,Float32Array);L.cams={n:camPos.length/3,pos:buf(camPos)};
// ---- math
function persp(f,a,n,r){const t=1/Math.tan(f/2),q=1/(n-r);return[t/a,0,0,0,0,t,0,0,0,0,(r+n)*q,-1,0,0,2*r*n*q,0]}
function look(e,c,u){let z=[e[0]-c[0],e[1]-c[1],e[2]-c[2]],l=Math.hypot(...z);z=z.map(x=>x/l);
 let x=[u[1]*z[2]-u[2]*z[1],u[2]*z[0]-u[0]*z[2],u[0]*z[1]-u[1]*z[0]];l=Math.hypot(...x);x=x.map(v=>v/l);
 const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
 return[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]),-(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]),-(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]),1]}
function mul(a,b){const o=new Array(16).fill(0);for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)o[j*4+i]+=a[k*4+i]*b[j*4+k];return o}
// ---- camera (orbit) state
const V={front:[0,.08,2.3],corner:[.85,.22,2.3],side:[1.5,.08,2.3],top:[0,1.5,2.6]};
let th=0,ph=.08,dist=2.3,tg=[0,0,0],spin=false,dark=false;
function setView(n){[th,ph,dist]=V[n];tg=[0,0,0]}
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setView(b.dataset.v));
document.getElementById('spin').onclick=()=>spin=!spin;
document.getElementById('bg').onclick=()=>dark=!dark;
setView('front');th=.35;
// ---- input
const pt=new Map();let last=null;
cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);pt.set(e.pointerId,e);last=null});
cv.addEventListener('pointerup',e=>{pt.delete(e.pointerId)});cv.addEventListener('pointercancel',e=>{pt.delete(e.pointerId)});
cv.addEventListener('contextmenu',e=>e.preventDefault());
function pan(dx,dy){const s=dist*.0016,c=Math.cos(th),n=Math.sin(th);
 tg[0]-=(c*dx)*s;tg[2]+=(n*dx)*s;tg[1]+=dy*s}
cv.addEventListener('pointermove',e=>{
 if(!pt.has(e.pointerId))return;const p=pt.get(e.pointerId);
 if(pt.size===2){const o=[...pt.values()].find(q=>q.pointerId!==e.pointerId),d0=Math.hypot(p.clientX-o.clientX,p.clientY-o.clientY),
  d1=Math.hypot(e.clientX-o.clientX,e.clientY-o.clientY);dist=Math.min(8,Math.max(.3,dist*d0/Math.max(d1,1)))}
 else{const dx=e.clientX-p.clientX,dy=e.clientY-p.clientY;
  if(e.buttons&2||e.shiftKey)pan(dx,dy);else{th-=dx*.006;ph=Math.max(-1.55,Math.min(1.55,ph+dy*.006))}}
 pt.set(e.pointerId,e)});
cv.addEventListener('wheel',e=>{e.preventDefault();dist=Math.min(8,Math.max(.3,dist*Math.exp(e.deltaY*.001)))},{passive:false});
// ---- draw
function draw(){
 const dpr=Math.min(devicePixelRatio||1,2),w=cv.clientWidth*dpr|0,h=cv.clientHeight*dpr|0;
 if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}
 gl.viewport(0,0,w,h);const bgc=dark?[.09,.09,.09]:[.96,.96,.95];gl.clearColor(...bgc,1);gl.enable(gl.DEPTH_TEST);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 if(spin)th+=.004;
 const eye=[tg[0]+dist*Math.cos(ph)*Math.sin(th),tg[1]+dist*Math.sin(ph),tg[2]+dist*Math.cos(ph)*Math.cos(th)];
 const m=mul(persp(.9,w/h,.02,50),look(eye,tg,[0,1,0])),size=+document.getElementById('ps').value*dpr;
 gl.useProgram(P1);gl.uniformMatrix4fv(gl.getUniformLocation(P1,'m'),false,m);gl.uniform1f(gl.getUniformLocation(P1,'s'),size);
 for(const k of ['sparse','dense']){if(!document.getElementById(k==='dense'?'cDense':'cSparse').checked)continue;
  const a=gl.getAttribLocation(P1,'a'),c=gl.getAttribLocation(P1,'c');
  gl.bindBuffer(gl.ARRAY_BUFFER,L[k].pos);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,L[k].col);gl.enableVertexAttribArray(c);gl.vertexAttribPointer(c,3,gl.UNSIGNED_BYTE,true,0,0);
  gl.drawArrays(gl.POINTS,0,L[k].n)}
 if(document.getElementById('cCams').checked){gl.useProgram(P2);gl.uniformMatrix4fv(gl.getUniformLocation(P2,'m'),false,m);
  gl.uniform3f(gl.getUniformLocation(P2,'k'),.85,.1,.1);const a=gl.getAttribLocation(P2,'a');
  gl.bindBuffer(gl.ARRAY_BUFFER,L.cams.pos);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);
  gl.drawArrays(gl.LINES,0,L.cams.n)}
 requestAnimationFrame(draw)}
document.getElementById('stats').textContent=DATA.stats;
draw();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results/dense")
    ap.add_argument("--rec", default="results/cache/rec_south-building_40_16_1.pkl")
    ap.add_argument("--out", default="results/viewer/south_building.html")
    ap.add_argument("--max_dense", type=int, default=200000)
    ap.add_argument("--dense_name", default="dense.ply")
    ap.add_argument("--crop_orbit", type=float, default=None, help="keep only points within this fraction of the orbit radius of the orbit axis")
    ap.add_argument("--axial", type=float, default=0.8, help="with --crop_orbit: half-height of the kept cylinder, as a fraction of the orbit radius")
    ap.add_argument("--title", default="South Building - multi-view 3D reconstruction")
    a = ap.parse_args()

    sp, sc = read_ply(os.path.join(a.folder, "sparse.ply"))
    dpath = os.path.join(a.folder, a.dense_name)
    dp, dc = read_ply(dpath) if os.path.exists(dpath) else (sp, sc)     # sets without a dense stage show the sparse cloud
    rec = pickle.load(open(a.rec, "rb"))

    if a.crop_orbit:      # object-centred crop: cylinder around the axis of the camera orbit (fit circle to the camera centres)
        Cc = np.array([G.camera_center(*rec.cams[i]) for i in sorted(rec.cams)])
        mu = Cc.mean(0)
        _, _, Vt = np.linalg.svd(Cc - mu)
        P2 = (Cc - mu) @ Vt[:2].T
        A_ = np.c_[2 * P2, np.ones(len(P2))]
        sol = np.linalg.lstsq(A_, (P2 ** 2).sum(1), rcond=None)[0]
        c2, R_orb = sol[:2], float(np.sqrt(sol[2] + sol[:2] @ sol[:2]))

        def _in_cyl(P):
            Q = P - mu
            return (np.linalg.norm(Q @ Vt[:2].T - c2, axis=1) < a.crop_orbit * R_orb) & (np.abs(Q @ Vt[2]) < a.axial * R_orb)
        kd, ks = _in_cyl(dp), _in_cyl(sp)
        dp, dc, sp, sc = dp[kd], dc[kd], sp[ks], sc[ks]
        print(f"object crop: orbit radius {R_orb:.2f}; kept {int(kd.sum()):,} dense / {int(ks.sum()):,} sparse points")
    md, ms = viz.clean_mask(dp, 3.0), viz.clean_mask(sp, 3.0)
    dp, dc, sp, sc = dp[md], dc[md], sp[ms], sc[ms]
    if len(dp) > a.max_dense:
        sel = np.random.default_rng(0).choice(len(dp), a.max_dense, replace=False)
        dp, dc = dp[sel], dc[sel]

    M = level_rotation(rec)
    c, r = viz.robust_bounds(dp @ M.T, lo=1, hi=99, pad=0.0)
    norm = lambda P: (((np.asarray(P) @ M.T) - c) / r).astype(np.float32)

    w, h = rec.K[0, 2] * 2, rec.K[1, 2] * 2
    seg = []
    for i in rec.cams:
        segs, _, _ = viz._frustum_lines(*rec.cams[i], rec.K, 0.12 * viz.robust_bounds(dp)[1], w, h)
        for p0, p1 in segs:
            seg += [p0, p1]
    cam_lines = norm(np.array(seg))

    stats = (f"{len(dp):,} dense points · {len(sp):,} sparse points · {len(rec.cams)} cameras. "
             f"Coloured from the photos; dense = rectified SGBM fused over 15 stereo pairs.")
    data = dict(dense=dict(n=len(dp), pos=b64(norm(dp)), col=b64(dc)),
                sparse=dict(n=len(sp), pos=b64(norm(sp)), col=b64(sc)),
                cams=dict(pos=b64(cam_lines)), stats=stats)
    html = (TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__TITLE__", a.title)
            .replace("__SUB__", "CT3205 Assignment 2 - own incremental SfM + dense stereo"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB): {len(dp):,} dense, {len(sp):,} sparse, {len(rec.cams)} cameras")


if __name__ == "__main__":
    main()
