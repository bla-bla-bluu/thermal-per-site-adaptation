"""Batch 9g: cost figures for the paper, measured when the GPU is otherwise idle.
  teacherMs   sliced-inference teacher (7 passes + merge), mean time per frame over 200 frames of s17 and s106
  cacheMB     size of the cached teacher detections of the 23 pools
  trainGB     peak GPU memory of a student run (Ultralytics' GPU_mem column of the training log, maximum over runs)
  trainMin    median / maximum wall time of the main students (STm), four runs sharing the GPU
  nRuns       finished training runs of the journal (runs/, runs_rtsfod/, runs_rtsfod_noada/)
Writes results/b9_cost.json."""
import os, glob, json, time, re
import numpy as np
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
SC = [s for s, *_ in scenarios()]; out = {}
ts = []
for sid in (17, 106):
    d = f"/tmp/b9_cost_{sid}"; os.makedirs(f"{d}/images", exist_ok=True)
    for p in sorted(glob.glob(f"{sdir(sid, 'test')}/images/*.jpg"))[:100]:
        q = f"{d}/images/{os.path.basename(p)}"
        if not os.path.exists(q): os.symlink(p, q)
    E.gen_sahi(BASELINE, 320, img_dir=f"{d}/images")                       # warm-up
    t0 = time.time(); r = E.gen_sahi(BASELINE, 320, img_dir=f"{d}/images"); ts.append((time.time() - t0) / len(r))
out["teacherMs"] = 1000 * float(np.mean(ts))
out["cacheMB"] = sum(os.path.getsize(f"{J}/cache/teacher_{tag(s)}_pool.pkl") for s in SC) / 2 ** 20
mem = []
for f in glob.glob(f"{J}/logs/train_STm_*.log"):
    mem += [float(x) for x in re.findall(r"\s(\d+(?:\.\d+)?)G\s", open(f, errors="ignore").read())]
out["trainGB"] = max(mem) if mem else float("nan")
dur = []
for d in glob.glob(f"{J}/runs/STm_*"):
    if os.path.exists(f"{d}/DONE"): dur.append((os.path.getmtime(f"{d}/DONE") - os.path.getmtime(f"{d}/args.yaml")) / 60)
out["trainMin"], out["trainMinMax"] = float(np.median(dur)), float(np.max(dur))
out["nRuns"] = sum(len(glob.glob(f"{J}/{r}/*/DONE")) for r in ("runs", "runs_rtsfod", "runs_rtsfod_noada"))
json.dump(out, open(f"{J}/results/b9_cost.json", "w"), indent=1); print(out)
