"""Batch 3: pseudo-label variants from the cached teacher detections (no new SAHI runs), then students.
Variants per scene: tau 0.25, 0.35, 0.55, 0.65 (with 0.45 from batch 1: a full threshold sweep);
'pers' = tau 0.45 minus static boxes (same place, IoU>=0.5, in >=80% of the frames t-10..t+10).
Also seeds 1 and 2 of the tau 0.45 student and of the supervised reference."""
import os, json, pickle, time, subprocess
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
def gmm_threshold(scores, lo=0.2, hi=0.7):
    x = np.asarray(scores, float); x = x[x >= 0.05]
    if len(x) < 10: return 0.45
    mu = np.array([np.percentile(x, 25), np.percentile(x, 90)]); sd = np.array([x.std(), x.std()]) + 1e-3; w = np.array([0.5, 0.5])
    for _ in range(200):
        p = w * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / sd; r = p / p.sum(1, keepdims=True)
        w = r.mean(0); mu = (r * x[:, None]).sum(0) / r.sum(0); sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / r.sum(0)) + 1e-3
    hi_c = int(np.argmax(mu)); g = np.linspace(0.05, 0.95, 181)
    p = w * np.exp(-0.5 * ((g[:, None] - mu) / sd) ** 2) / sd; post = p[:, hi_c] / p.sum(1)
    idx = np.where((post >= 0.5) & (g >= mu.min()))[0]
    return float(np.clip(g[idx[0]] if len(idx) else 0.45, lo, hi))
def persistent(D, t, k, W=10, ps=0.8):
    nb = [D[u] for u in range(max(0, t - W), min(len(D), t + W + 1)) if u != t]
    if not nb: return False
    return np.mean([len(x) > 0 and E.iou_mat(D[t][k:k+1], x).max() >= 0.5 for x in nb]) >= ps
if __name__ != "__main__":
    raise SystemExit  # run as a script only
info = {}
for sid, name, split, imgs, labs in scenarios():
    rec = sorted(pickle.load(open(f"{J}/cache/teacher_s{sid:02d}_pool.pkl", "rb")), key=lambda r: r["im"])
    taus = {"t025": 0.25, "t035": 0.35, "t055": 0.55, "t065": 0.65, "pers": 0.45}
    info[sid] = {}
    D45 = [r["det"][r["conf"] >= 0.45] for r in rec]
    for var, tau in taus.items():
        d = f"{J}/data/flc/s{sid:02d}/pl_{var}"; os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
        nb = 0
        for t, r in enumerate(rec):
            src = os.path.realpath(f"{sdir(sid, 'pool')}/images/{r['im']}")
            if not os.path.islink(f"{d}/images/{r['im']}"): os.symlink(src, f"{d}/images/{r['im']}")
            W, H = map(float, Image.open(src).size)
            if var == "pers":
                B = [D45[t][k] for k in range(len(D45[t])) if not persistent(D45, t, k)]
            else:
                B = list(r["det"][r["conf"] >= tau])
            nb += len(B)
            open(f"{d}/labels/{os.path.splitext(r['im'])[0]}.txt", "w").write("".join(
                f"0 {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n" for b in B))
        open(f"{J}/data/flc/s{sid:02d}_{var}.yaml", "w").write(f"path: {d}\ntrain: images\nval: {sdir(sid, 'val')}/images\nnames:\n  0: person\n")
        info[sid][var] = nb / len(rec)
    print(sid, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info[sid].items()}, flush=True)
json.dump(info, open(f"{J}/results/batch3_pseudolabel_info.json", "w"), indent=1)
PY = "/opt/conda/bin/python"; running = []
def launch(name, data, init=COCO_N, seed=0):
    if os.path.exists(f"{J}/runs/{name}/DONE"): return
    code = (f"from ultralytics import YOLO; YOLO('{init}').train(data='{data}', epochs=50, imgsz=640, batch=16, device='cuda:0', "
            f"workers=4, cache='ram', project='{J}/runs', name='{name}', exist_ok=True, patience=50, seed={seed}, plots=False); "
            f"open('{J}/runs/{name}/DONE','w').write('ok')")
    p = subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J)
    running.append((name, p))
def wait(n):
    while True:
        running[:] = [(a, p) for a, p in running if p.poll() is None]
        if len(running) < n: return
        time.sleep(15)
jobs = []
for sid, name, split, imgs, labs in scenarios():
    for var in ("t025", "t035", "t055", "t065", "pers"):
        jobs.append((f"ST_s{sid:02d}_{var}_seed0", f"{J}/data/flc/s{sid:02d}_{var}.yaml", 0))
    for seed in (1, 2):
        jobs.append((f"ST_s{sid:02d}_t045_seed{seed}", f"{J}/data/flc/s{sid:02d}_st045.yaml", seed))
        jobs.append((f"SUP_s{sid:02d}_seed{seed}", f"{J}/data/flc/s{sid:02d}_sup.yaml", seed))
for name, data, seed in jobs:
    wait(3); launch(name, data, seed=seed)
wait(1)
def cached(key, fn):
    p = f"{J}/cache/{key}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    v = fn(); pickle.dump(v, open(p, "wb")); return v
out = open(f"{J}/results/flc_models.jsonl", "a")
for name, data, seed in jobs:
    sid = int(name.split("_")[1][1:]); w = f"{J}/runs/{name}/weights/last.pt"
    if not os.path.exists(w): print("missing", name); continue
    rec, _ = cached(f"{name}_last_test_std", lambda: E.gen_std(w, f"{J}/data/flc/s{sid:02d}_test.yaml"))
    dets, gts = E.match(rec); m = E.summarize(dets, len(gts)); m["frames"] = len(rec)
    out.write(json.dumps({"sid": sid, "model": name, "mode": "std", **m}) + "\n"); out.flush()
print("BATCH3_DONE")
