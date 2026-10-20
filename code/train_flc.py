"""Per-scenario training, at most 3 runs at once on the one GPU:
  ST  : self-training student on the tau=0.45 pseudo-labels of the pool (COCO-pretrained YOLOv8n, as in ICVGIP),
        per-epoch checkpoints kept for the label-free model-selection study
  SUP : supervised in-domain reference on the same pool frames with their real labels
Recipe: 50 epochs, 640 px, batch 16 (pools are 133-805 frames), AdamW auto, final epoch, seed 0, RAM cache."""
import os, time, subprocess
from jcommon import *
PY = "/opt/conda/bin/python"; running = []
def launch(name, data, extra=""):
    if os.path.exists(f"{J}/runs/{name}/DONE"):
        return
    code = (f"from ultralytics import YOLO; YOLO('{COCO_N}').train(data='{data}', epochs=50, imgsz=640, batch=16, device='cuda:0', "
            f"workers=4, cache='ram', project='{J}/runs', name='{name}', exist_ok=True, patience=50, seed=0, plots=False{extra}); "
            f"open('{J}/runs/{name}/DONE','w').write('ok')")
    os.makedirs(f"{J}/logs", exist_ok=True)
    p = subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J)
    running.append((name, p)); print("launched", name, flush=True)
def wait(n):
    while True:
        running[:] = [(a, p) for a, p in running if p.poll() is None]
        if len(running) < n: return
        time.sleep(15)
for sid, name, split, imgs, labs in scenarios():
    wait(3); launch(f"ST_s{sid:02d}_t045_seed0", f"{J}/data/flc/s{sid:02d}_st045.yaml", ", save_period=1")
    wait(3); launch(f"SUP_s{sid:02d}_seed0", f"{J}/data/flc/s{sid:02d}_sup.yaml")
wait(1)
print("TRAIN_DONE")
