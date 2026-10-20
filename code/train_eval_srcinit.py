"""Batch 2b: students initialised from the urban baseline (fine-tuning on the pseudo-labels) instead of COCO weights;
same pseudo-labels (tau 0.45) and recipe. Then test evaluation (std and SAHI), final epoch."""
import os, json, pickle, time, subprocess
from jcommon import *
import cr_eval as E
PY = "/opt/conda/bin/python"; running = []
def launch(name, data):
    if os.path.exists(f"{J}/runs/{name}/DONE"): return
    code = (f"from ultralytics import YOLO; YOLO('{BASELINE}').train(data='{data}', epochs=50, imgsz=640, batch=16, device='cuda:0', "
            f"workers=4, cache='ram', project='{J}/runs', name='{name}', exist_ok=True, patience=50, seed=0, plots=False); "
            f"open('{J}/runs/{name}/DONE','w').write('ok')")
    p = subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J)
    running.append((name, p)); print("launched", name, flush=True)
def wait(n):
    while True:
        running[:] = [(a, p) for a, p in running if p.poll() is None]
        if len(running) < n: return
        time.sleep(15)
sc = scenarios()
for sid, name, split, imgs, labs in sc:
    wait(2); launch(f"STsrc_s{sid:02d}_t045_seed0", f"{J}/data/flc/s{sid:02d}_st045.yaml")
wait(1)
def cached(key, fn):
    p = f"{J}/cache/{key}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    v = fn(); pickle.dump(v, open(p, "wb")); return v
out = open(f"{J}/results/flc_models.jsonl", "a")
for sid, name, split, imgs, labs in sc:
    w = f"{J}/runs/STsrc_s{sid:02d}_t045_seed0/weights/last.pt"
    for mode in ("std", "sahi"):
        key = f"STsrc-last_s{sid:02d}_test_{mode}"
        rec = cached(key, lambda: E.gen_std(w, f"{J}/data/flc/s{sid:02d}_test.yaml")[0] if mode == "std" else E.gen_sahi(w, 320, img_dir=f"{sdir(sid, 'test')}/images"))
        dets, gts = E.match(rec); m = E.summarize(dets, len(gts)); m["CI"] = E.bootstrap(dets, gts, len(rec), n=2000); m["frames"] = len(rec)
        out.write(json.dumps({"sid": sid, "name": name, "model": "STsrc-last", "mode": mode, **m}) + "\n"); out.flush()
        print(sid, name, "STsrc-last", mode, f"AP50 {100*m['AP50']:.1f}", flush=True)
print("SRCINIT_DONE")
