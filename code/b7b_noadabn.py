"""Batch 7b-ablation: RT-SFOD stage 2 (official settings, 60 epochs, 640 px) started from the YOLO26n source model
itself, i.e. without its AdaBN warm start, 3 seeds x 24 scenes; then per-epoch val/test predictions are cached as
cache/b7/rtsfodnb_<scene>_seed<k>.pkl and the epoch checkpoints deleted (as b7_epochs.py does).
Usage: b7b_noadabn.py train   (launches at most 3 runs at once)   |   b7b_noadabn.py cache"""
import os, sys, time, glob, re, pickle, subprocess
import numpy as np
from jcommon import *
SRC26 = f"{J}/runs/J_Base_y26n/weights/last.pt"; RS = f"{J}/baselines/RT-SFOD"; PYR = "/path/to/envs/rtsfod/bin/python"
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
jobs = [(sid, seed) for seed in (0, 1, 2) for sid, *_ in scenarios()]
if sys.argv[1] == "train":
    running = []
    def wait(n):
        while True:
            running[:] = [p for p in running if p.poll() is None]
            if len(running) < n: return
            time.sleep(15)
    for sid, seed in jobs:
        out = f"{J}/runs_rtsfod_noada/{tag(sid)}_seed{seed}"
        if os.path.exists(f"{out}/DONE"): continue
        os.makedirs(out, exist_ok=True); wait(3)
        cmd = (f"cd {RS} && {PYR} scripts/YOLO26/stage2_rtsfod_yolo26.py --stage1_model {SRC26} --data {J}/data/flc/{tag(sid)}_rtsfod.yaml "
               f"--out_dir {out} --imgsz 640 --batch 16 --workers 4 --epochs 60 --seed {seed} --device 0 && touch {out}/DONE")
        running.append(subprocess.Popen(["bash", "-c", cmd], stdout=open(f"{J}/logs/rtsfodnb_{tag(sid)}_seed{seed}.log", "w"), stderr=subprocess.STDOUT))
        print("launched", sid, seed, flush=True)
    wait(1); print("B7B_TRAIN_DONE", flush=True)
else:
    from PIL import Image
    from ultralytics import YOLO
    from selection_study_helpers import predict
    for sid, seed in jobs:
        run = f"{J}/runs_rtsfod_noada/{tag(sid)}_seed{seed}"; cf = f"{J}/cache/b7/rtsfodnb_{tag(sid)}_seed{seed}.pkl"
        cks = {int(re.search(r"epoch_(\d+)\.pt", p).group(1)): p for p in glob.glob(f"{run}/checkpoints/*epoch_*.pt")}
        if os.path.exists(cf) or not os.path.exists(f"{run}/DONE") or not cks: continue
        out = {}
        for part in ("val", "test"):
            arr = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in sorted(glob.glob(f"{sdir(sid, part)}/images/*.jpg"))]
            out[part] = {"ep": {e: predict(YOLO(cks[e]), arr) for e in sorted(cks)}}
        pickle.dump(out, open(cf, "wb"))
        for p in cks.values(): os.remove(p)
        print("cached", sid, seed, len(cks), flush=True)
    print("B7B_CACHE_DONE", flush=True)
