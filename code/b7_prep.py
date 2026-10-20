"""Batch 7a (same-detector comparison with RT-SFOD, which is built on YOLO26):
YOLO26n source model (LLVIP, runs/J_Base_y26n) -> baseline predictions on each scene's val/test frames (cached),
SAHI teacher on the pool (cached), tau=0.55 pseudo-labels for YOLO26n students, and RT-SFOD data yamls."""
import os, glob, pickle
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
from selection_study_helpers import predict
SRC26 = f"{J}/runs/J_Base_y26n/weights/last.pt"
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
os.makedirs(f"{J}/cache/b7", exist_ok=True); m = YOLO(SRC26)
for sid, *_ in scenarios():
    t = tag(sid); cf = f"{J}/cache/b7/base26_{t}.pkl"
    if not os.path.exists(cf):
        out = {p: predict(m, [np.asarray(Image.open(q).convert("RGB"))[:, :, ::-1] for q in sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg"))])
               for p in ("val", "test")}
        pickle.dump(out, open(cf, "wb"))
    tf = f"{J}/cache/b7/teacher26_{t}_pool.pkl"
    rec = pickle.load(open(tf, "rb")) if os.path.exists(tf) else E.gen_sahi(SRC26, 320, img_dir=f"{sdir(sid, 'pool')}/images")
    pickle.dump(rec, open(tf, "wb"))
    d = sdir(sid, "pl26_t055"); os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True); nb = 0
    for r in rec:
        src = os.path.realpath(f"{sdir(sid, 'pool')}/images/{r['im']}")
        if not os.path.islink(f"{d}/images/{r['im']}"): os.symlink(src, f"{d}/images/{r['im']}")
        W, H = map(float, Image.open(src).size); B = r["det"][r["conf"] >= 0.55]; nb += len(B)
        open(f"{d}/labels/{os.path.splitext(r['im'])[0]}.txt", "w").write("".join(
            f"0 {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n" for b in B))
    open(f"{J}/data/flc/{t}_y26t055.yaml", "w").write(f"path: {d}\ntrain: images\nval: {sdir(sid, 'val')}/images\nnames:\n  0: person\n")
    open(f"{J}/data/flc/{t}_rtsfod.yaml", "w").write(f"path: {sdir(sid, 'pool')}\ntrain: images\nval: {sdir(sid, 'val')}/images\nnames:\n  0: person\n")
    print(sid, "teacher boxes/frame at 0.55:", round(nb / max(len(rec), 1), 2), flush=True)
print("B7_PREP_DONE")
