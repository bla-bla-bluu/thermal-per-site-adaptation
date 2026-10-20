"""Batch 10a: confidence calibration of the teacher, the source detector and the students.
A detection is correct if the per-frame greedy matching (descending confidence, IoU >= 0.5) assigns it to a person.
Per recording and model: reliability bins of width 0.1 over detections scoring >= 0.1 (count, mean confidence, share
correct) and the detection expected calibration error D-ECE = sum_b n_b/N |correct_b - conf_b| (Kueppers et al. 2020).
  teacher   sliced-inference detections on the labelled pool frames (what the pseudo-label threshold acts on)
  theta0    source detector on the test frames
  final     final-epoch tau=0.55 students (STm, three seeds) on the test frames
Writes results/b10_calib.json."""
import os, json, pickle, glob
import numpy as np
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
EDGES = np.round(np.arange(0.1, 1.0001, 0.1), 2)
def correct_flags(G, D, C):
    conf, ok = [], []
    for g, d, c in zip(G, D, C):
        o = np.argsort(-c, kind="stable"); d, c = d[o], c[o]; used = np.zeros(len(g), bool)
        M = E.iou_mat(d, g) if len(d) and len(g) else np.zeros((len(d), len(g)))
        for k in range(len(d)):
            hit = 0
            if len(g):
                row = np.where(used, -1, M[k]); j = int(np.argmax(row))
                if row[j] >= 0.5: used[j] = True; hit = 1
            conf.append(c[k]); ok.append(hit)
    return np.array(conf), np.array(ok)
def bins(conf, ok):
    m = conf >= EDGES[0]; conf, ok = conf[m], ok[m]; out = []; ece = 0.0
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        k = (conf >= a) & ((conf < b) if b < 1 else (conf <= b))
        n = int(k.sum()); out.append([float(a), n, float(conf[k].mean()) if n else None, float(ok[k].mean()) if n else None])
        if n: ece += n * abs(ok[k].mean() - conf[k].mean())
    return {"bins": out, "dece": float(ece / max(len(conf), 1)), "n": int(len(conf))}
res = {}
for sid, *_ in scenarios():
    t = tag(sid); r = {}
    rec = sorted(pickle.load(open(f"{J}/cache/teacher_{t}_pool.pkl", "rb")), key=lambda x: x["im"])
    lab = {f[:-4] for f in os.listdir(f"{sdir(sid, 'pool')}/labels") if f.endswith(".txt")}
    rec = [x for x in rec if os.path.splitext(x["im"])[0] in lab]
    r["teacher"] = bins(*correct_flags([x["gt"] for x in rec], [x["det"] for x in rec], [x["conf"] for x in rec]))
    gt = pickle.load(open(f"{J}/cache/b9/gt_{t}_test.pkl", "rb")); b = pickle.load(open(f"{J}/cache/b6/baseline_{t}.pkl", "rb"))["test"]
    r["theta0"] = bins(*correct_flags(gt, [d for d, _ in b], [c for _, c in b]))
    cs, os_ = [], []
    for k in range(3):
        st = pickle.load(open(f"{J}/cache/b9/STm_{t}_seed{k}.pkl", "rb"))["test"][50]
        c, o = correct_flags(gt, [d for d, _ in st], [c for _, c in st]); cs.append(c); os_.append(o)
    r["final"] = bins(np.concatenate(cs), np.concatenate(os_))
    res[t] = r; print(t, {k: round(v["dece"], 3) for k, v in r.items()}, flush=True)
json.dump(res, open(f"{J}/results/b10_calib.json", "w"), indent=1)
print("B10_CALIB_DONE")
