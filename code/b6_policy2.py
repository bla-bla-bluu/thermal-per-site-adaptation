"""Batch 6d: treat 'do not adapt' as one more candidate in few-label model selection. On the same n labelled val
frames, compare the urban baseline with the student's epochs and keep whichever scores best; report its test AP50.
Baseline val/test predictions come from YOLO.predict like the student's, so the comparison is like for like."""
import os, json, pickle, glob
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
    d, g = E.match(rec); return E.summarize(d, len(g))["AP50"]
bm = YOLO(BASELINE); BP = {}
for sid, *_ in scenarios():
    cf = f"{J}/cache/b6/baseline_{tag(sid)}.pkl"
    if os.path.exists(cf): BP[sid] = pickle.load(open(cf, "rb")); continue
    out = {}
    for part in ("val", "test"):
        ps = sorted(glob.glob(f"{sdir(sid, part)}/images/*.jpg"))
        out[part] = predict(bm, [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps])
    pickle.dump(out, open(cf, "wb")); BP[sid] = out
rng = np.random.default_rng(0); rows = {}
for sid, *_ in scenarios():
    per = {k: [] for k in ("base", "final", "pick5", "pick10", "pick20", "pick5+base", "pick10+base", "pick20+base", "opt")}
    for seed in (0, 1, 2):
        c = pickle.load(open(f"{J}/cache/b6/{tag(sid)}_seed{seed}.pkl", "rb")); eps = sorted(c["test"]["ep"])
        vg, tg = c["val"]["gt"], c["test"]["gt"]; nv = len(vg)
        t = {e: 100 * ap(tg, c["test"]["ep"][e]) for e in eps}; tb = 100 * ap(tg, BP[sid]["test"])
        per["base"].append(tb); per["final"].append(t[eps[-1]]); per["opt"].append(max(tb, max(t.values())))
        for n in (5, 10, 20):
            a, b = [], []
            for _ in range(50 if n < nv else 1):
                idx = rng.choice(nv, min(n, nv), replace=False)
                sc = {e: ap(vg, c["val"]["ep"][e], idx) for e in eps}
                e_best = max(eps, key=lambda e: (sc[e], -e)); a.append(t[e_best])
                b.append(t[e_best] if sc[e_best] > ap(vg, BP[sid]["val"], idx) else tb)
            per[f"pick{n}"].append(np.mean(a)); per[f"pick{n}+base"].append(np.mean(b))
    rows[sid] = {k: float(np.mean(v)) for k, v in per.items()}
B = np.array([rows[s]["base"] for s in rows])
for k in ("base", "final", "pick5", "pick5+base", "pick10", "pick10+base", "pick20", "pick20+base", "opt"):
    v = np.array([rows[s][k] for s in rows]); d = v - B
    print(f"{k:12s} mean AP50 {v.mean():5.1f} | helps {int((d > 2).sum()):2d} hurts {int((d < -2).sum()):2d} | worst {d.min():6.1f}")
json.dump(rows, open(f"{J}/results/b6_policy2.json", "w"), indent=1)
print("scenes", len(rows))
