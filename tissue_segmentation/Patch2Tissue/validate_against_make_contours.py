#!/usr/bin/env python3
"""
Validate patch2tissue_tasknode against the implementation it ports,
scripts_patched/make_contours.py, on that script's own paired input/output set.

    inputs : /project/zhihuanglab/songhao/lnco2/results_zeroshot/tumor/data/*.zarr
             (Patch-Classification: class_indices, coordinates, classes/*)
    truth  : /project/zhihuanglab/songhao/lnco2/results_zeroshot/tumor/contour/*.zarr
             (Region-Contours: contours, offsets, is_hole, areas)
    slides : /project/zhihuanglab/songhao/lnco2/slides/*.ndpi   (for level-0 dimensions)

Usage:
    /home/tissuelab-admin/.conda/envs/musk/bin/python validate_against_make_contours.py [N]

    N = how many slides to check (default: all 107). Expect 107/107 exact,
    IoU 1.0000. Anything less means a regression -- see README > Validation for
    the one caveat that produced 0.9995.
"""
import os, sys, glob, warnings, statistics as st
import numpy as np, zarr, cv2
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch2tissue_tasknode as N
R="/project/zhihuanglab/songhao/lnco2/results_zeroshot/tumor"
K_CLOSE = K_OPEN = 3   # make_contours.py 默认 --connect-patches 3

def ref(c):
    g=zarr.open_group(c,mode="r")["Region-Contours"]
    pts,offs,hole,ar=g["contours"][:],g["offsets"][:],g["is_hole"][:],g["areas"][:]
    polys=[pts[offs[i]:offs[i+1]] for i in range(len(offs)-1)]
    return polys, hole, ar
SLIDE_DIR="/project/zhihuanglab/songhao/lnco2/slides"
def slide_dims(d, coords):
    base=os.path.basename(d)[:-5]          # strip .zarr -> AI-....ndpi
    sp=os.path.join(SLIDE_DIR, base)
    if os.path.exists(sp):
        try:
            import tiffslide
            sl=tiffslide.TiffSlide(sp); w,h=sl.dimensions; sl.close(); return int(w),int(h)
        except Exception: pass
    return int(coords[:,2].max()), int(coords[:,3].max())

def mine(d):
    zf=zarr.open_group(d,mode="r"); dd=N.read_patch_inputs(zf)
    coords, ci = dd["coords"], dd["class_indices"]
    patch=int(np.median(coords[:,2]-coords[:,0])) or 1
    members=np.where(ci>=1)[0]                 # 与 make_contours 的 labels>=1 同义
    if len(members)==0: return [], np.zeros(0,np.uint8), np.zeros(0,np.int64)
    W0,H0 = slide_dims(d, coords)
    grid,_,_=N.build_patch_grid(coords,members,patch,W0,H0)
    grid=N.grid_close_open(grid,K_CLOSE,K_OPEN)
    return N.contours_from_grid(grid,patch,0.0)
def iou(A_,B_,S=224):
    A_=[np.asarray(p,dtype=np.float64) for p in A_]; B_=[np.asarray(p,dtype=np.float64) for p in B_]
    if not A_ and not B_: return 1.0
    if not A_ or not B_: return 0.0
    W=max(int(p[:,0].max()) for p in A_+B_); H=max(int(p[:,1].max()) for p in A_+B_)
    A=np.zeros((H//S+2,W//S+2),np.uint8); B=A.copy()
    for p in A_: cv2.fillPoly(A,[(p/S).astype(np.int32)],1)
    for p in B_: cv2.fillPoly(B,[(p/S).astype(np.int32)],1)
    u=int((A|B).sum()); return int((A&B).sum())/u if u else float('nan')

ds=sorted(glob.glob(f"{R}/data/*.zarr"))
lim=int(sys.argv[1]) if len(sys.argv)>1 else len(ds)
rows=[]; exact=0
for d in ds[:lim]:
    b=os.path.basename(d); c=f"{R}/contour/{b}"
    if not os.path.exists(c): continue
    rp,rh,ra=ref(c); mp,mh,ma=mine(d)
    ro=[p for p,h in zip(rp,rh) if not h]; mo=[p for p,h in zip(mp,mh) if not h]
    v=iou(ro,mo)
    same = (len(rp)==len(mp)) and (len(ro)==len(mo)) and int(ra[rh==0].sum())==int(ma[mh==0].sum())
    exact += same
    rows.append((len(ro),len(mo),int(rh.sum()),int(mh.sum()),
                 int(ra[rh==0].sum()) if len(ra) else 0, int(ma[mh==0].sum()) if len(ma) else 0, v, same))
print(f"{'slide':22s}{'ref_r':>6s}{'my_r':>6s}{'ref_h':>6s}{'my_h':>6s}{'area_ratio':>11s}{'IoU':>7s}{'exact':>7s}")
print("-"*72)
for (d,r) in list(zip(ds[:lim],rows))[:12]:
    b=os.path.basename(d).split(".")[0]
    print(f"{b:22s}{r[0]:>6d}{r[1]:>6d}{r[2]:>6d}{r[3]:>6d}{(r[5]/r[4] if r[4] else float('nan')):>11.3f}{r[6]:>7.3f}{str(r[7]):>7s}")
print("-"*72)
print(f"共 {len(rows)} 张")
print(f"逐位完全一致(轮廓数+孔数+总面积): {exact}/{len(rows)}")
print(f"区域数完全相同: {sum(1 for r in rows if r[0]==r[1])}/{len(rows)}")
print(f"IoU 中位 {st.median(r[6] for r in rows):.4f} | 均值 {st.mean(r[6] for r in rows):.4f} | 最小 {min(r[6] for r in rows):.4f}")
nz=[r for r in rows if r[4]]
if nz: print(f"面积比 中位 {st.median(r[5]/r[4] for r in nz):.4f} | 最小 {min(r[5]/r[4] for r in nz):.4f} | 最大 {max(r[5]/r[4] for r in nz):.4f}")
