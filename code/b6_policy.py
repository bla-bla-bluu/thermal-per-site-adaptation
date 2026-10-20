"""Batch 6c: the full 'safe self-training' policy over 24 scenes x 3 seeds, all on the tau=0.55 students.
  never adapt          the urban baseline
  always, final epoch  self-training as in ICVGIP
  always, n-frame pick checkpoint chosen on n random labelled val frames (50 draws), n in {5, 10, 20}
  go/no-go + 10-frame  adapt only if the baseline-teacher agreement signal passes a threshold fitted on the other
                       scenes (leave one scene out); then pick the checkpoint on 10 labelled frames
  oracle               best of baseline and the test-best epoch (unattainable)
Test AP50 from the same matcher and AP routine as everything else (YOLO.predict pipeline for per-epoch values)."""
import os, json, pickle, glob, collections
import numpy as np
from jcommon import *
import cr_eval as E
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def ap(gt, det, idx=None):
    idx = range(len(gt)) if idx is None else idx
    rec = [{"gt": gt[i], "det": det[i][0], "conf": det[i][1], "gain": 1.0} for i in idx]
    d, g = E.match(rec); return E.summarize(d, len(g))["AP50"]
base = {}
for l in open(f"{J}/results/teacher_baseline.jsonl"):
    r = json.loads(l)
    if r.get("model") == "Baseline" and r["mode"] == "std": base[r["sid"]] = 100 * r["AP50"]
# baseline-teacher agreement on the pool (label-free): AP50 of the baseline's boxes against the teacher's tau 0.45 boxes
sig = {}
for sid, *_ in scenarios():
    rec = sorted(pickle.load(open(f"{J}/cache/teacher_{tag(sid)}_pool.pkl", "rb")), key=lambda r: r["im"])
    bp = pickle.load(open(f"{J}/cache/Baseline_{tag(sid)}_pool_predict.pkl", "rb"))
    sig[sid] = 100 * ap([r["det"][r["conf"] >= 0.45] for r in rec], bp)
rng = np.random.default_rng(0); R = collections.defaultdict(dict)
for sid, *_ in scenarios():
    for seed in (0, 1, 2):
        cf = f"{J}/cache/b6/{tag(sid)}_seed{seed}.pkl"
        if not os.path.exists(cf): continue
        c = pickle.load(open(cf, "rb")); eps = sorted(c["test"]["ep"])
        t = {e: 100 * ap(c["test"]["gt"], c["test"]["ep"][e]) for e in eps}
        v = c["val"]; nv = len(v["gt"]); row = {"final": t[eps[-1]], "test_opt": max(t.values())}
        for n in (5, 10, 20):
            picks = []
            for _ in range(50 if n < nv else 1):
                idx = rng.choice(nv, min(n, nv), replace=False)
                best = max(eps, key=lambda e: (ap(v["gt"], v["ep"][e], idx), -e)); picks.append(t[best])
            row[f"pick{n}"] = float(np.mean(picks))
        R[sid][seed] = row
sids = [s for s in sorted(R) if len(R[s]) == 3]
M = lambda k: np.array([np.mean([R[s][sd][k] for sd in (0, 1, 2)]) for s in sids])
B = np.array([base[s] for s in sids]); X = np.array([sig[s] for s in sids])
pol = {"never adapt": B, "always, final epoch": M("final"), "always, 5-frame pick": M("pick5"),
       "always, 10-frame pick": M("pick10"), "always, 20-frame pick": M("pick20")}
gng = []
for i in range(len(sids)):
    tr = [j for j in range(len(sids)) if j != i]; best = (-1, None, None)
    for thr in np.unique(X[tr]):
        for side in (1, -1):
            a = np.mean([M("pick10")[j] if side * (X[j] - thr) >= 0 else B[j] for j in tr])
            if a > best[0]: best = (a, thr, side)
    _, thr, side = best; gng.append(M("pick10")[i] if side * (X[i] - thr) >= 0 else B[i])
pol["go/no-go + 10-frame pick"] = np.array(gng)
pol["oracle (best of baseline / test-best epoch)"] = np.maximum(B, M("test_opt"))
for name, v in pol.items():
    d = v - B
    print(f"{name:44s} mean AP50 {v.mean():5.1f} | helps {int((d > 2).sum()):2d} hurts {int((d < -2).sum()):2d} | worst {d.min():6.1f}")
print("scenes:", len(sids))
