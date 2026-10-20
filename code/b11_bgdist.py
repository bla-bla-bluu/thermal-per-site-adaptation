"""Diagnostic: distribution of the background-difference fraction (cache/b11/fg_<tag>.pkl) for teacher boxes (tau 0.55)
that match a person (TP) and that do not (FP), on the labelled pool frames."""
import os, pickle, glob, numpy as np
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
import json
OUTJ = {}
for f in sorted(glob.glob(f"{J}/cache/b11/fg_*.pkl")):
    t = os.path.basename(f)[3:-4]; out = pickle.load(open(f, "rb")); tp, fp = [], []
    for r in out:
        if len(r["gt"]) == 0 and not r["im"]: continue
        sel = r["conf"] >= 0.55; D = r["det"][sel]; F = r["fg"][sel]; G = r["gt"]
        if len(D) == 0: continue
        if len(G) == 0: fp += list(F); continue
        M = E.iou_mat(D, G); used = np.zeros(len(G), bool)
        for k in range(len(D)):
            row = np.where(used, -1, M[k]); j = int(np.argmax(row))
            if row[j] >= 0.5: used[j] = True; tp.append(F[k])
            else: fp.append(F[k])
    q = lambda a: np.percentile(a, [10, 25, 50, 75]).round(2) if len(a) else "-"
    print(t, "TP n=%d" % len(tp), q(tp), "| FP n=%d" % len(fp), q(fp))
    OUTJ[t] = {"tp_n": len(tp), "fp_n": len(fp), "tp_med": float(np.median(tp)) if len(tp) else None, "fp_med": float(np.median(fp)) if len(fp) else None}
json.dump(OUTJ, open(f"{J}/results/b11_bgdist.json", "w"), indent=1)
