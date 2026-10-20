"""Batch 6a: students on the tau=0.55 pseudo-labels (the best fixed threshold over 24 scenes) with every epoch saved,
three seeds, all 24 scenes (JDOMAIN=all). Recipe as before (COCO YOLOv8n, 50 epochs, 640 px, batch 16)."""
import os, time, subprocess
from jcommon import *
PY = "/opt/conda/bin/python"; running = []
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def launch(name, data, seed):
    if os.path.exists(f"{J}/runs/{name}/DONE"): return
    code = (f"from ultralytics import YOLO; YOLO('{COCO_N}').train(data='{data}', epochs=50, imgsz=640, batch=16, device='cuda:0', "
            f"workers=4, cache='ram', project='{J}/runs', name='{name}', exist_ok=True, patience=50, seed={seed}, plots=False, save_period=1); "
            f"open('{J}/runs/{name}/DONE','w').write('ok')")
    p = subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J)
    running.append((name, p)); print("launched", name, flush=True)
def wait(n):
    while True:
        running[:] = [(a, p) for a, p in running if p.poll() is None]
        if len(running) < n: return
        time.sleep(15)
for seed in (0, 1, 2):
    for sid, name, split, imgs, labs in scenarios():
        wait(3); launch(f"STe_{tag(sid)}_t055_seed{seed}", f"{J}/data/flc/{tag(sid)}_t055.yaml", seed)
wait(1)
print("B6_TRAIN_DONE")
