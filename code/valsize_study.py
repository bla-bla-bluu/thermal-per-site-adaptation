"""How many labelled target frames does checkpoint selection need? For every self-training run (tau 0.45, seed 0)
and every epoch: detections on the val frames (labelled, 10% of the scene). Offline, for n in {3, 5, 10, 20, 40, all}
labelled frames (random frames from the val window, or one contiguous clip of n frames), pick the epoch with the best
AP50 on those n frames and look up its test AP50 (from the selection study). 50 draws per (scene, n)."""
import os, sys, json, glob, pickle
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
from selection_study_helpers import load_gt, predict
os.makedirs(f"{J}/cache/valsize", exist_ok=True)
for sid, name, split, imgs, labs in scenarios():
    cf = f"{J}/cache/valsize/s{sid:02d}.pkl"
    if os.path.exists(cf): continue
    run = f"{J}/runs/ST_s{sid:02d}_t045_seed0"; vp = sorted(glob.glob(f"{sdir(sid, 'val')}/images/*.jpg"))
    va = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in vp]; out = {"gt": load_gt(vp), "ep": {}}
    for ep in range(50):
        ck = f"{run}/weights/epoch{ep}.pt"
        if os.path.exists(ck): out["ep"][ep + 1] = predict(YOLO(ck), va)
    pickle.dump(out, open(cf, "wb")); print("cached", sid, name, flush=True)
test = {}
for f in glob.glob(f"{J}/results/selection_w*.jsonl"):
    for l in open(f):
        r = json.loads(l); test[(r["sid"], r["epoch"])] = 100 * r["test_AP50"]
rng = np.random.default_rng(0); NS = (3, 5, 10, 20, 40, None); rows = {}
def ap_on(gt, det, idx):
    rec = [{"gt": gt[i], "det": det[i][0], "conf": det[i][1], "gain": 1.0} for i in idx]
    d, g = E.match(rec); return E.summarize(d, len(g))["AP50"]
for sid, name, split, imgs, labs in scenarios():
    c = pickle.load(open(f"{J}/cache/valsize/s{sid:02d}.pkl", "rb")); nv = len(c["gt"]); eps = sorted(c["ep"])
    for mode in ("random", "contiguous"):
        for n in NS:
            k = nv if n is None or n >= nv else n; vals = []
            for _ in range(50 if k < nv else 1):
                if mode == "random": idx = rng.choice(nv, k, replace=False)
                else: s0 = rng.integers(0, nv - k + 1); idx = np.arange(s0, s0 + k)
                best = max(eps, key=lambda e: (ap_on(c["gt"], c["ep"][e], idx), -e))
                vals.append(test[(sid, best)])
            rows[(sid, mode, n)] = float(np.mean(vals))
    print(sid, {f"{m[0]}{n}": round(rows[(sid, m, n)], 1) for m in ("random", "contiguous") for n in NS}, flush=True)
for mode in ("random", "contiguous"):
    print(mode, {("all" if n is None else n): round(float(np.mean([rows[(s, mode, n)] for s, *_ in scenarios()])), 1) for n in NS})
print("final epoch", round(float(np.mean([test[(s, 50)] for s, *_ in scenarios()])), 1))
print("VALSIZE_DONE")
