"""Batch 11a: background gating of the teacher's pseudo-labels, a label-free remedy for the
warm-clutter failure. The cameras are fixed, so a background image can be estimated from the unlabelled pool itself;
warm walls, brick stacks and machinery belong to it, people walking through do not. A pseudo-label is kept only if
enough of its pixels differ from the background. Settings fixed before looking at any result:
  background   per-pixel median of the pool frames within +-W pool frames (W = 100), at half resolution,
               computed at anchors every 20 frames (each frame uses its nearest anchor; handles camera moves and drift)
  difference   frame fitted to its background by least squares (gain and offset, second pass without the 5% largest
               residuals, for automatic gain changes), foreground = |residual| > 3 robust sd (1.4826 MAD)
  gate         keep a teacher box (tau = 0.55) if >= 10% of its pixels are foreground
This script only measures pseudo-label precision and recall on the labelled pool frames, with and without the gate
(diagnostics, as for the other variants), plus the per-recording share of gated-out boxes (a label-free signal).
Writes cache/b11/fg_<tag>.pkl (per pool frame: half-resolution foreground fraction per teacher box) and
results/b11_bgcheck.json."""
import os, sys, json, pickle, glob
import numpy as np
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
W, STEP, K, FMIN, TAU = 100, 20, 3.0, 0.10, 0.55
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
os.makedirs(f"{J}/cache/b11", exist_ok=True)
def fgfrac(sid):
    t = tag(sid); rec = sorted(pickle.load(open(f"{J}/cache/teacher_{t}_pool.pkl", "rb")), key=lambda r: r["im"])
    ims = [np.asarray(Image.open(f"{sdir(sid, 'pool')}/images/{r['im']}").convert("L"), dtype=np.float32)[::2, ::2] for r in rec]
    A = np.stack(ims); n = len(A); anchors = list(range(0, n, STEP)); bgs = {}
    for a in anchors: bgs[a] = np.median(A[max(0, a - W):min(n, a + W + 1)], axis=0)
    out = []
    for i, r in enumerate(rec):
        bg = bgs[min(anchors, key=lambda a: abs(a - i))]; f = A[i]
        X = np.stack([bg.ravel(), np.ones(bg.size)], 1); y = f.ravel()
        c = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ c
        keep = np.abs(res) <= np.percentile(np.abs(res), 95); c = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        res = (y - X @ c).reshape(bg.shape); sd = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        m = np.abs(res) > K * sd
        o = np.argsort(-r["conf"], kind="stable"); d = r["det"][o]; s = r["conf"][o]
        fr = []
        for b in d:
            x1, y1, x2, y2 = [int(round(v / 2)) for v in b]; x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(m.shape[1], max(x2, x1 + 1)), min(m.shape[0], max(y2, y1 + 1))
            fr.append(float(m[y1:y2, x1:x2].mean()) if (x2 > x1 and y2 > y1) else 0.0)
        out.append({"im": r["im"], "det": d, "conf": s, "fg": np.array(fr), "gt": r["gt"]})
    pickle.dump(out, open(f"{J}/cache/b11/fg_{t}.pkl", "wb")); return out
def quality(out, labelled, gate):
    tp = nb = ng = 0
    for r in out:
        if os.path.splitext(r["im"])[0] not in labelled: continue
        sel = r["conf"] >= TAU
        if gate: sel &= r["fg"] >= FMIN
        D = r["det"][sel]; G = r["gt"]; nb += len(D); ng += len(G)
        if len(D) and len(G):
            M = E.iou_mat(D, G); used = np.zeros(len(G), bool)
            for k in range(len(D)):
                row = np.where(used, -1, M[k]); j = int(np.argmax(row))
                if row[j] >= 0.5: used[j] = True; tp += 1
    return {"boxes": nb, "P": tp / max(nb, 1), "R": tp / max(ng, 1)}
res = {}
for sid, *_ in scenarios():
    t = tag(sid); out = fgfrac(sid)
    labelled = {f[:-4] for f in os.listdir(f"{sdir(sid, 'pool')}/labels") if f.endswith(".txt")}
    allb = np.concatenate([r["fg"][r["conf"] >= TAU] for r in out]) if out else np.zeros(0)
    res[t] = {"t055": quality(out, labelled, False), "bg": quality(out, labelled, True),
              "gated_share": float((allb < FMIN).mean()) if len(allb) else 0.0, "teacher_boxes": int(len(allb))}
    q0, q1 = res[t]["t055"], res[t]["bg"]
    print(t, "P %.2f->%.2f R %.2f->%.2f gated %.2f of %d boxes" % (q0["P"], q1["P"], q0["R"], q1["R"], res[t]["gated_share"], len(allb)), flush=True)
json.dump(res, open(f"{J}/results/b11_bgcheck.json", "w"), indent=1)
print("B11_BGCHECK_DONE")
