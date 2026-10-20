"""Batch 9d: per-epoch predictions of the batch-9 students, cached in cache/b9/<prefix>_<tag>_seed<k>.pkl
as {part: {epoch: [(boxes, conf)]}}, then the epoch checkpoints are deleted (last.pt and best.pt stay). Parts: val and test;
STm also gets the gap frames and their mirror images (label-free checkpoint scores). Ground truth goes to
cache/b9/gt_<tag>_<part>.pkl. Usage: b9_epochs.py <worker> <n_workers>; polls until runs/B9_TRAIN_DONE exists and every
finished run is cached."""
import os, sys, glob, pickle, time
import numpy as np
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
from ultralytics import YOLO
from selection_study_helpers import load_gt, predict
wk, nw = int(sys.argv[1]), int(sys.argv[2]); os.makedirs(f"{J}/cache/b9", exist_ok=True)
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
SC = [s for s, *_ in scenarios()]
jobs = [(pre, s, seed) for pre in ("STm", "STsrc", "STne", "STns", "STpe") for seed in (0, 1, 2) for s in SC]
mine = [j for n, j in enumerate(jobs) if n % nw == wk]; ARR = {}
def arrays(sid, part):
    if (sid, part) not in ARR:
        ps = sorted(glob.glob(f"{sdir(sid, 'gap' if part == 'gapflip' else part)}/images/*.jpg"))
        a = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
        ARR[(sid, part)] = [np.ascontiguousarray(x[:, ::-1]) for x in a] if part == "gapflip" else a
        gf = f"{J}/cache/b9/gt_{tag(sid)}_{part}.pkl"
        if part in ("val", "test") and not os.path.exists(gf):
            pickle.dump(load_gt(ps), open(gf + f".tmp{wk}", "wb")); os.replace(gf + f".tmp{wk}", gf)
    return ARR[(sid, part)]
while True:
    train_done = os.path.exists(f"{J}/runs/B9_TRAIN_DONE"); pending = 0
    for pre, sid, seed in mine:
        run = f"{J}/runs/{pre}_{tag(sid)}_t055_seed{seed}"; cf = f"{J}/cache/b9/{pre}_{tag(sid)}_seed{seed}.pkl"
        if os.path.exists(cf): continue
        if not os.path.exists(f"{run}/DONE"):
            pending += not train_done; continue          # after the queue has finished, a run without DONE has failed
        parts = ("val", "test", "gap", "gapflip") if pre == "STm" else ("val", "test")
        out = {p: {} for p in parts}
        for ep in range(50):
            ck = f"{run}/weights/epoch{ep}.pt"
            if not os.path.exists(ck): continue
            m = YOLO(ck)
            for p in parts: out[p][ep + 1] = predict(m, arrays(sid, p))
        if pre == "STm":     # mirrored gap boxes back to the original frame coordinates
            W = arrays(sid, "gap")[0].shape[1]
            out["gapflip"] = {e: [(np.stack([W - d[:, 2], d[:, 1], W - d[:, 0], d[:, 3]], 1) if len(d) else d, c) for d, c in v]
                              for e, v in out["gapflip"].items()}
        pickle.dump(out, open(cf, "wb"))
        for ck in glob.glob(f"{run}/weights/epoch*.pt"): os.remove(ck)
        print(time.strftime("%H:%M:%S"), "cached", pre, sid, seed, len(out["test"]), flush=True)
    if train_done and pending == 0: break
    time.sleep(60)
print("B9_EPOCHS_DONE", wk, flush=True)
