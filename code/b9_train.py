"""Batch 9c: training queue. Every student: YOLOv8n, 50 epochs, 640 px, batch 16, the
recipe of the paper, seeds 0-2. Runs marked keep save every epoch (save_period=1); b9_epochs.py caches their per-epoch
predictions and deletes the epoch checkpoints.
  STm    main configuration again (COCO init, tau 0.55), so that val, test AND gap predictions of every epoch exist
  STsrc  student initialised from the source detector (tau 0.55)
  STne   frames without pseudo-labels left out (tau 0.55)
  STns   teacher without slicing (tau 0.55)
  STpe   static boxes removed (tau 0.55)
  ST_<tag>_{t025,t035,t065}_seed{1,2}  threshold sweep, seeds 1 and 2 (final epoch only; seed 0 exists from batch 3)
Usage: b9_train.py [parallel runs, default 4]. Writes runs/B9_TRAIN_DONE at the end."""
import os, sys, time, subprocess
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
PY = "/opt/conda/bin/python"; NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 4; running = []
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def launch(name, data, init, seed, keep):
    if os.path.exists(f"{J}/runs/{name}/DONE"): return
    sp = ", save_period=1" if keep else ""
    code = (f"from ultralytics import YOLO; YOLO('{init}').train(data='{data}', epochs=50, imgsz=640, batch=16, device='cuda:0', "
            f"workers=4, cache='ram', project='{J}/runs', name='{name}', exist_ok=True, patience=50, seed={seed}, plots=False{sp}); "
            f"open('{J}/runs/{name}/DONE','w').write('ok')")
    p = subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J)
    running.append((name, p)); print(time.strftime("%H:%M:%S"), "launched", name, flush=True)
def wait(n):
    while True:
        running[:] = [(a, p) for a, p in running if p.poll() is None]
        if len(running) < n: return
        time.sleep(10)
SC = [s for s, *_ in scenarios()]; jobs = []
for pre, var, init in (("STm", "t055", COCO_N), ("STsrc", "t055", BASELINE), ("STne", "t055ne", COCO_N),
                       ("STns", "ns055", COCO_N), ("STpe", "pers055", COCO_N)):
    for seed in (0, 1, 2):
        jobs += [(f"{pre}_{tag(s)}_t055_seed{seed}", f"{J}/data/flc/{tag(s)}_{var}.yaml", init, seed, True) for s in SC]
for var in ("t025", "t035", "t065"):
    for seed in (1, 2):
        jobs += [(f"ST_{tag(s)}_{var}_seed{seed}", f"{J}/data/flc/{tag(s)}_{var}.yaml", COCO_N, seed, False) for s in SC]
for job in jobs:
    wait(NPAR); launch(*job)
wait(1)
open(f"{J}/runs/B9_TRAIN_DONE", "w").write("ok"); print("B9_TRAIN_DONE", flush=True)
