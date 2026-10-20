"""Batch 6b: per-epoch predictions of every tau=0.55 student on its scene's val and test frames (cached), then the
epoch checkpoints are deleted to save disk (last.pt and best.pt stay). Usage: b6_epochs.py <worker> <n_workers>"""
import os, sys, glob, pickle
import numpy as np
from PIL import Image
from jcommon import *
from ultralytics import YOLO
from selection_study_helpers import load_gt, predict
wk, nw = int(sys.argv[1]), int(sys.argv[2]); os.makedirs(f"{J}/cache/b6", exist_ok=True)
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
jobs = [(sid, seed) for seed in (0, 1, 2) for sid, *_ in scenarios()]
for n, (sid, seed) in enumerate(jobs):
    if n % nw != wk: continue
    run = f"{J}/runs/STe_{tag(sid)}_t055_seed{seed}"; cf = f"{J}/cache/b6/{tag(sid)}_seed{seed}.pkl"
    if os.path.exists(cf) or not os.path.exists(f"{run}/DONE"): continue
    out = {}
    for part in ("val", "test"):
        ps = sorted(glob.glob(f"{sdir(sid, part)}/images/*.jpg"))
        out[part] = {"gt": load_gt(ps), "arr": [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps], "ep": {}}
    for ep in range(50):
        ck = f"{run}/weights/epoch{ep}.pt"
        if not os.path.exists(ck): continue
        m = YOLO(ck)
        for part in ("val", "test"):
            out[part]["ep"][ep + 1] = predict(m, out[part]["arr"])
    for part in ("val", "test"):
        del out[part]["arr"]
    pickle.dump(out, open(cf, "wb"))
    for ck in glob.glob(f"{run}/weights/epoch*.pt"):
        os.remove(ck)
    print("cached", sid, seed, flush=True)
print("B6_EPOCHS_DONE", wk)
