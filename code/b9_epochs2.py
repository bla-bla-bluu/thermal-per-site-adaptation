"""Batch 9d, second version (replaces b9_epochs.py while the queue runs): the same per-epoch caching, but any worker
takes the next finished, unclaimed run (a lock file per run), so more workers can share the backlog; results are
written to a temporary file and renamed. Usage: b9_epochs2.py <worker id>; writes logs/B9_EPOCHS2_DONE_<id> at the end."""
import os, sys, glob, pickle, time
import numpy as np
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
from ultralytics import YOLO
from selection_study_helpers import load_gt, predict
wk = sys.argv[1]; os.makedirs(f"{J}/cache/b9/locks", exist_ok=True)
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
SC = [s for s, *_ in scenarios()]
jobs = [(pre, s, seed) for pre in ("STm", "STsrc", "STne", "STns", "STpe") for seed in (0, 1, 2) for s in SC]
ARR = {}
def arrays(sid, part):
    if (sid, part) not in ARR:
        if len(ARR) > 40: ARR.clear()
        ps = sorted(glob.glob(f"{sdir(sid, 'gap' if part == 'gapflip' else part)}/images/*.jpg"))
        a = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
        ARR[(sid, part)] = [np.ascontiguousarray(x[:, ::-1]) for x in a] if part == "gapflip" else a
        gf = f"{J}/cache/b9/gt_{tag(sid)}_{part}.pkl"
        if part in ("val", "test") and not os.path.exists(gf):
            pickle.dump(load_gt(ps), open(gf + f".tmp{wk}", "wb")); os.replace(gf + f".tmp{wk}", gf)
    return ARR[(sid, part)]
while True:
    train_done = os.path.exists(f"{J}/runs/B9_TRAIN_DONE"); pending = 0
    for pre, sid, seed in jobs:
        run = f"{J}/runs/{pre}_{tag(sid)}_t055_seed{seed}"; cf = f"{J}/cache/b9/{pre}_{tag(sid)}_seed{seed}.pkl"
        if os.path.exists(cf): continue
        if not os.path.exists(f"{run}/DONE"):
            pending += not train_done; continue
        try: fd = os.open(f"{J}/cache/b9/locks/{pre}_{tag(sid)}_seed{seed}", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError: continue
        parts = ("val", "test", "gap", "gapflip") if pre == "STm" else ("val", "test")
        out = {p: {} for p in parts}
        for ep in range(50):
            ck = f"{run}/weights/epoch{ep}.pt"
            if not os.path.exists(ck): continue
            m = YOLO(ck)
            for p in parts: out[p][ep + 1] = predict(m, arrays(sid, p))
        if pre == "STm":
            W = arrays(sid, "gap")[0].shape[1]
            out["gapflip"] = {e: [(np.stack([W - d[:, 2], d[:, 1], W - d[:, 0], d[:, 3]], 1) if len(d) else d, c) for d, c in v]
                              for e, v in out["gapflip"].items()}
        pickle.dump(out, open(cf + f".tmp{wk}", "wb")); os.replace(cf + f".tmp{wk}", cf)
        for ck in glob.glob(f"{run}/weights/epoch*.pt"): os.remove(ck)
        print(time.strftime("%H:%M:%S"), "cached", pre, sid, seed, len(out["test"]), flush=True)
    if train_done and pending == 0:
        # every finished run is cached or claimed; wait for claims held by other workers to finish
        left = [1 for pre, sid, seed in jobs if os.path.exists(f"{J}/runs/{pre}_{tag(sid)}_t055_seed{seed}/DONE")
                and not os.path.exists(f"{J}/cache/b9/{pre}_{tag(sid)}_seed{seed}.pkl")]
        if not left: break
    time.sleep(30)
open(f"{J}/logs/B9_EPOCHS2_DONE_{wk}", "w").write("ok"); print("B9_EPOCHS2_DONE", wk, flush=True)
