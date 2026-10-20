"""SF-UT at the official number of passes: what the final checkpoint looks like. Per recording and seed (cache/sfut_epoch):
AP50 of the source model, of the final checkpoint and of the best checkpoint; people per test frame; boxes with score >= 0.5 per frame
at the final checkpoint and for the source model; mean per-frame maximum score; first checkpoint with AP50 below half the source AP50.
Writes results/b15_collapse.json."""
import os, sys, json, glob, numpy as np
os.environ.setdefault("JDOMAIN", "all"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jcommon import *
from b9_policy import fmatch, ap_idx, L
def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1]); x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None); aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)
def recur(dets, thr=0.5, lag=5):
    """Share of the boxes with score >= thr that have an IoU >= 0.5 box with score >= thr in the frame `lag` test frames later."""
    r = []
    for i in range(len(dets) - lag):
        (b0, s0), (b1, s1) = dets[i], dets[i + lag]; m0, m1 = s0 >= thr, s1 >= thr
        if m0.sum() and m1.sum(): r.append(float((iou(b0[m0], b1[m1]).max(1) >= 0.5).mean()))
    return float(np.mean(r)) if r else None
def stats(dets, thr=0.5):
    return float(np.mean([int((cf >= thr).sum()) for _, cf in dets])), float(np.mean([cf.max() if len(cf) else 0 for _, cf in dets]))
res = {}
for f in sorted(glob.glob(f"{J}/cache/sfut_epoch/*_seed*.pkl")):
    t, seed = os.path.basename(f)[:-4].split("_seed"); seed = int(seed)
    gt = L(f"{J}/cache/b9/gt_{t}_test.pkl"); nt = len(gt); b = L(f"{J}/cache/sfut/base_{t}.pkl")["test"]; c = L(f); eps = sorted(c["test"])
    base = ap_idx(*fmatch(gt, b), np.arange(nt)); ta = [ap_idx(*fmatch(gt, c["test"][e]), np.arange(nt)) for e in eps]
    n5b, mxb = stats(b); n5f, mxf = stats(c["test"][eps[-1]])
    first = next((i + 1 for i, a in enumerate(ta) if a < base / 2), None)
    res.setdefault(t, {})[seed] = dict(gt=sum(len(g) for g in gt) / nt, base=base, final=ta[-1], best=max(ta), n5_base=n5b, mx_base=mxb, n5_final=n5f, mx_final=mxf, recur_final=recur(c["test"][eps[-1]]),
                                       first_half=first, ntest=nt, ncheck=len(eps), curve=[round(float(a), 2) for a in ta])
    print(t, seed, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[t][seed].items() if k != "curve"}, flush=True)
json.dump(res, open(f"{J}/results/b15_collapse.json", "w"), indent=1)
