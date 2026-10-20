"""Batch 7d: same policies for the YOLO26n detectors: baseline (no adaptation), our self-training and RT-SFOD,
each at the final epoch and with the checkpoint picked on 10 / 20 labelled val frames (50 draws), 3 seeds, 24 scenes."""
import os, glob, pickle
import numpy as np
from jcommon import *
import cr_eval as E
from selection_study_helpers import load_gt
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def ap(gt, det, idx=None):
    idx = range(len(gt)) if idx is None else idx
    rec = [{"gt": gt[i], "det": det[i][0], "conf": det[i][1], "gain": 1.0} for i in idx]
    d, g = E.match(rec); return E.summarize(d, len(g))["AP50"]
rng = np.random.default_rng(0); res = {}
for sid, *_ in scenarios():
    gt = {p: load_gt(sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg"))) for p in ("val", "test")}
    base = pickle.load(open(f"{J}/cache/b7/base26_{tag(sid)}.pkl", "rb")); row = {"base26": 100 * ap(gt["test"], base["test"])}
    for meth in ("st26", "rtsfod"):
        per = {"final": [], "pick10": [], "pick20": []}
        for seed in (0, 1, 2):
            cf = f"{J}/cache/b7/{meth}_{tag(sid)}_seed{seed}.pkl"
            if not os.path.exists(cf): continue
            c = pickle.load(open(cf, "rb")); eps = sorted(c["test"]["ep"]); nv = len(gt["val"])
            t = {e: 100 * ap(gt["test"], c["test"]["ep"][e]) for e in eps}; per["final"].append(t[eps[-1]])
            for n in (10, 20):
                picks = []
                for _ in range(50 if n < nv else 1):
                    idx = rng.choice(nv, min(n, nv), replace=False)
                    picks.append(t[max(eps, key=lambda e: (ap(gt["val"], c["val"]["ep"][e], idx), -e))])
                per[f"pick{n}"].append(np.mean(picks))
        for k, v in per.items():
            if v: row[f"{meth}_{k}"] = float(np.mean(v))
    res[sid] = row; print(sid, {k: round(v, 1) for k, v in row.items()}, flush=True)
keys = sorted({k for r in res.values() for k in r})
B = np.array([res[s]["base26"] for s in res])
for k in keys:
    v = np.array([res[s].get(k, np.nan) for s in res]); d = v - B
    print(f"{k:16s} mean AP50 {np.nanmean(v):5.1f} | helps {int(np.nansum(d > 2)):2d} hurts {int(np.nansum(d < -2)):2d} | n={int(np.sum(~np.isnan(v)))}")
