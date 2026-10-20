"""B4: can label-free signals predict whether self-training will help a scene? Scene-level signals (no target labels):
  density      teacher boxes per pool frame (tau 0.45)
  t_temporal   share of teacher boxes matched (IoU>=0.3) in the adjacent pool frame
  t_static     share of teacher boxes present at the same place (IoU>=0.5) in >=80% of frames t-10..t+10
  hiconf       share of teacher boxes (>=0.45) scoring >=0.65
  base_teacher AP50 of the baseline's standard-inference boxes against the teacher's boxes on the pool frames
  s_agree, s_flip, s_temporal  final-epoch student signals on the gap frames (selection study)
Outcome: gain = self-training (final epoch, mean over available seeds) minus baseline, test AP50.
Reports Spearman correlations and a leave-one-scene-out go/no-go rule on each single signal."""
import os, json, glob, pickle, collections
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
def spearman(a, b):
    a, b = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(a, b)[0, 1])
res = collections.defaultdict(dict)
for f in ("teacher_baseline.jsonl", "flc_models.jsonl"):
    for l in open(f"{J}/results/{f}"):
        r = json.loads(l)
        if "model" in r: res[r["sid"]][(r["model"], r["mode"])] = 100 * r["AP50"]
sel = collections.defaultdict(dict)
for f in glob.glob(f"{J}/results/selection_w*.jsonl"):
    for l in open(f):
        r = json.loads(l); sel[r["sid"]][r["epoch"]] = r
base_model = YOLO(BASELINE); sig = {}
for sid, name, split, imgs, labs in scenarios():
    rec = sorted(pickle.load(open(f"{J}/cache/teacher_s{sid:02d}_pool.pkl", "rb")), key=lambda r: r["im"])
    D = [r["det"][r["conf"] >= 0.45] for r in rec]; C = [r["conf"][r["conf"] >= 0.45] for r in rec]
    nb = sum(len(d) for d in D); hit = tot = 0; stat = 0
    for t in range(len(D)):
        for u in (t - 1, t + 1):
            if 0 <= u < len(D) and len(D[t]):
                tot += len(D[t]); hit += int((E.iou_mat(D[t], D[u]).max(1) >= 0.3).sum()) if len(D[u]) else 0
        nbh = [D[u] for u in range(max(0, t - 10), min(len(D), t + 11)) if u != t]
        for k in range(len(D[t])):
            if nbh and np.mean([len(x) > 0 and E.iou_mat(D[t][k:k+1], x).max() >= 0.5 for x in nbh]) >= 0.8: stat += 1
    cache = f"{J}/cache/Baseline_s{sid:02d}_pool_predict.pkl"
    if os.path.exists(cache): bp = pickle.load(open(cache, "rb"))
    else:
        bp = []
        for r in rec:
            o = base_model.predict(f"{sdir(sid, 'pool')}/images/{r['im']}", imgsz=640, conf=0.001, iou=0.5, verbose=False, device="cuda:0")[0]
            bp.append((o.boxes.xyxy.cpu().numpy().reshape(-1, 4), o.boxes.conf.cpu().numpy().reshape(-1)))
        pickle.dump(bp, open(cache, "wb"))
    m = E.summarize(*[(lambda d, g: (d, len(g)))(*E.match([{"gt": g, "det": d, "conf": c, "gain": 1.0} for g, (d, c) in zip(D, bp)]))][0])
    S = sel.get(sid, {}); last = S.get(50, {})
    sig[sid] = {"density": nb / len(rec), "t_temporal": hit / max(tot, 1), "t_static": stat / max(nb, 1),
                "hiconf": float(np.mean(np.concatenate(C) >= 0.65)) if nb else 0.0, "base_teacher": 100 * m["AP50"],
                "s_agree": 100 * last.get("agree_AP50", np.nan), "s_flip": 100 * last.get("flip_AP50", np.nan), "s_temporal": last.get("temporal", np.nan)}
sids = sorted(sig)
st = {s: [res[s][k] for k in res[s] if k[1] == "std" and (k[0] == "ST-last" or k[0].startswith("ST_s") and "_t045_seed" in k[0])] for s in sids}
gain = np.array([np.mean(st[s]) - res[s][("Baseline", "std")] for s in sids]); base = np.array([res[s][("Baseline", "std")] for s in sids])
stv = np.array([np.mean(st[s]) for s in sids])
print("scenes", len(sids), "| seeds per scene:", sorted(set(len(st[s]) for s in sids)))
print(f"{'sid':>3} {'gain':>6} " + " ".join(f"{k:>12}" for k in sig[sids[0]]))
for i, s in enumerate(sids):
    print(f"{s:>3} {gain[i]:6.1f} " + " ".join(f"{v:12.2f}" for v in sig[s].values()))
print("\nSpearman(signal, gain):", {k: round(spearman([sig[s][k] for s in sids], gain), 2) for k in sig[sids[0]] if not np.isnan([sig[s][k] for s in sids]).any()})
print("\nLeave-one-scene-out go/no-go (adapt if signal on the right side of a threshold fitted on the other scenes):")
print(f"  never adapt {base.mean():.1f} | always adapt {stv.mean():.1f} | perfect choice {np.maximum(base, stv).mean():.1f}")
for k in sig[sids[0]]:
    x = np.array([sig[s][k] for s in sids])
    if np.isnan(x).any(): continue
    out = []
    for i in range(len(sids)):
        tr = [j for j in range(len(sids)) if j != i]; best = (-1, None, None)
        for thr in np.unique(x[tr]):
            for side in (1, -1):
                a = [stv[j] if side * (x[j] - thr) >= 0 else base[j] for j in tr]
                if np.mean(a) > best[0]: best = (np.mean(a), thr, side)
        _, thr, side = best; out.append(stv[i] if side * (x[i] - thr) >= 0 else base[i])
    print(f"  {k:12s} {np.mean(out):.1f}")
