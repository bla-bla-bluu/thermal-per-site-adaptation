"""Batch 7e: per-scene numbers for the YOLO26n comparison, saved as results/b7_policy2.json for the manuscript.
  base26          YOLO26n LLVIP source model, no adaptation
  adabn           RT-SFOD stage 1 only (AdaBN warm start, the model its stage 2 starts from)
  st26_* / rtsfod_* / rtsfodnb_* (RT-SFOD stage 2 without AdaBN, when cached)   final epoch, checkpoint picked on n in {10, 20} random labelled val frames (50 draws), the same
                  with the source model as one more candidate (+base), and the test-best epoch (opt); mean over 3 seeds.
Same matcher and AP routine as everything else; source/AdaBN predictions from YOLO.predict like the cached epochs."""
import os, glob, pickle, json
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
from selection_study_helpers import load_gt, predict
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def ap(gt, det, idx=None):
    idx = range(len(gt)) if idx is None else idx
    rec = [{"gt": gt[i], "det": det[i][0], "conf": det[i][1], "gain": 1.0} for i in idx]
    d, g = E.match(rec); return 100 * E.summarize(d, len(g))["AP50"]
rng = np.random.default_rng(0); res = {}
for sid, *_ in scenarios():
    t = tag(sid); gt = {p: load_gt(sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg"))) for p in ("val", "test")}
    base = pickle.load(open(f"{J}/cache/b7/base26_{t}.pkl", "rb")); nv = len(gt["val"])
    row = {"base26": ap(gt["test"], base["test"])}
    cf = f"{J}/cache/b7/adabn_{t}.pkl"
    if not os.path.exists(cf):
        ck = glob.glob(f"{J}/runs_rtsfod/{t}_stage1/*.pt")[0]; m = YOLO(ck)
        arr = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in sorted(glob.glob(f"{sdir(sid, 'test')}/images/*.jpg"))]
        pickle.dump({"test": predict(m, arr)}, open(cf, "wb"))
    row["adabn"] = ap(gt["test"], pickle.load(open(cf, "rb"))["test"])
    for meth in ("st26", "rtsfod", "rtsfodnb"):
        if not all(os.path.exists(f"{J}/cache/b7/{meth}_{t}_seed{k}.pkl") for k in (0, 1, 2)): continue
        per = {k: [] for k in ("final", "pick10", "pick20", "pick10+base", "pick20+base", "opt")}
        for seed in (0, 1, 2):
            c = pickle.load(open(f"{J}/cache/b7/{meth}_{t}_seed{seed}.pkl", "rb")); eps = sorted(c["test"]["ep"])
            tt = {e: ap(gt["test"], c["test"]["ep"][e]) for e in eps}
            per["final"].append(tt[eps[-1]]); per["opt"].append(max(tt.values()))
            for n in (10, 20):
                a, b = [], []
                for _ in range(50 if n < nv else 1):
                    idx = rng.choice(nv, min(n, nv), replace=False)
                    sc = {e: ap(gt["val"], c["val"]["ep"][e], idx) for e in eps}
                    e_best = max(eps, key=lambda e: (sc[e], -e)); a.append(tt[e_best])
                    b.append(tt[e_best] if sc[e_best] > ap(gt["val"], base["val"], idx) else row["base26"])
                per[f"pick{n}"].append(np.mean(a)); per[f"pick{n}+base"].append(np.mean(b))
        for k, v in per.items(): row[f"{meth}_{k}"] = float(np.mean(v))
    res[t] = row; print(t, {k: round(v, 1) for k, v in row.items()}, flush=True)
json.dump(res, open(f"{J}/results/b7_policy2.json", "w"), indent=1)
B = np.array([r["base26"] for r in res.values()])
for k in res[next(iter(res))]:
    if not all(k in r for r in res.values()): continue
    v = np.array([r[k] for r in res.values()]); d = v - B
    print(f"{k:18s} mean AP50 {v.mean():5.1f} | helps {int((d > 2).sum()):2d} hurts {int((d < -2).sum()):2d} | worst {d.min():6.1f}")
print("B7_POLICY2_DONE")
