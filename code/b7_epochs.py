"""Batch 7c: per-epoch predictions on val/test for the YOLO26n self-training students and the RT-SFOD students
(cached), then their epoch checkpoints are deleted. Usage: b7_epochs.py <method: st26|rtsfod> <worker> <n_workers>"""
import os, sys, glob, pickle, re
import numpy as np
from PIL import Image
from jcommon import *
from ultralytics import YOLO
from selection_study_helpers import predict
meth, wk, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
jobs = [(sid, seed) for seed in (0, 1, 2) for sid, *_ in scenarios()]
for n, (sid, seed) in enumerate(jobs):
    if n % nw != wk: continue
    if meth == "st26":
        run = f"{J}/runs/ST26_{tag(sid)}_t055_seed{seed}"; cks = {int(re.search(r"epoch(\d+)\.pt", p).group(1)) + 1: p for p in glob.glob(f"{run}/weights/epoch*.pt")}
    else:
        run = f"{J}/runs_rtsfod/{tag(sid)}_seed{seed}"; cks = {int(re.search(r"epoch_(\d+)\.pt", p).group(1)): p for p in glob.glob(f"{run}/checkpoints/*epoch_*.pt")}
    cf = f"{J}/cache/b7/{meth}_{tag(sid)}_seed{seed}.pkl"
    if os.path.exists(cf) or not os.path.exists(f"{run}/DONE") or not cks: continue
    out = {}
    for part in ("val", "test"):
        ps = sorted(glob.glob(f"{sdir(sid, part)}/images/*.jpg"))
        arr = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
        out[part] = {"ep": {e: predict(YOLO(cks[e]), arr) for e in sorted(cks)}}
    pickle.dump(out, open(cf, "wb"))
    for p in cks.values(): os.remove(p)
    print("cached", meth, sid, seed, len(cks), flush=True)
print("B7_EPOCHS_DONE", meth, wk)
