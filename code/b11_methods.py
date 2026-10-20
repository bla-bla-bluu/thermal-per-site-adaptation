"""Batch 11b, from the batch-9 per-epoch caches only (no training, no new predictions).
Candidate improvements, evaluated per recording and seed on the test window like the paper's policies:
  ens9     label-free: weighted box fusion of epochs 10, 15, ..., 50 (9 models; boxes >= 0.05; IoU 0.55; fused score =
           sum of member scores / 9); a trajectory ensemble instead of the final epoch
  ens10    label-free: the same for the last 10 epochs (41..50)
  dis{n}   few-label: the n validation frames on which the candidates (theta0 and the 50 epochs) disagree most (standard
           deviation of the number of boxes >= 0.25) are labelled, instead of n random frames; then the usual guard with
           theta0. Deterministic per seed. n = 3, 5, 10
  multi10  few-label: guard on the same 10 random frames as the paper, over theta0 and the epochs of four trajectories
           (main, empty frames left out, static boxes removed, source initialisation)
  margin{d} few-label: the paper's ten-frame guard, but a student replaces theta0 only if it beats theta0 by more than
           d AP50 points on the labelled frames (d = 2, 5)
Writes results/b11_methods.json."""
import os, json, pickle, numpy as np
from multiprocessing import Pool
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
from b9_policy import fmatch, ap_idx, draws, guard, L
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
SC = [s for s, *_ in scenarios()]
def wbf(dets, K, thr=0.55, smin=0.05):
    B = np.concatenate([d[c >= smin] for d, c in dets]) if dets else np.zeros((0, 4))
    S = np.concatenate([c[c >= smin] for d, c in dets]) if dets else np.zeros(0)
    if len(S) == 0: return np.zeros((0, 4)), np.zeros(0)
    o = np.argsort(-S); B, S = B[o], S[o]; FB, FS, W = [], [], []
    for b, s in zip(B, S):
        if FB:
            F = np.array(FB); ix1 = np.maximum(F[:, 0], b[0]); iy1 = np.maximum(F[:, 1], b[1]); ix2 = np.minimum(F[:, 2], b[2]); iy2 = np.minimum(F[:, 3], b[3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            iou = inter / ((F[:, 2] - F[:, 0]) * (F[:, 3] - F[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9); j = int(np.argmax(iou))
            if iou[j] >= thr:
                w = W[j] + s; FB[j] = (np.array(FB[j]) * W[j] + b * s) / w; W[j] = w; FS[j] += s; continue
        FB.append(b.astype(float)); FS.append(float(s)); W.append(float(s))
    return np.array(FB).reshape(-1, 4), np.array(FS) / K
def counts(det, thr=0.25): return np.array([int((c >= thr).sum()) for _, c in det])
def one(sid):
    t = tag(sid); gv, gt = L(f"{J}/cache/b9/gt_{t}_val.pkl"), L(f"{J}/cache/b9/gt_{t}_test.pkl"); nv, nt = len(gv), len(gt)
    bp = L(f"{J}/cache/b6/baseline_{t}.pkl"); fm0v, ngtv = fmatch(gv, bp["val"]); fm0t, ngtt = fmatch(gt, bp["test"])
    base = ap_idx(fm0t, ngtt, np.arange(nt)); R = {"base": base, "seeds": []}
    for seed in (0, 1, 2):
        C = {pre: L(f"{J}/cache/b9/{pre}_{t}_seed{seed}.pkl") for pre in ("STm", "STne", "STpe", "STsrc") if os.path.exists(f"{J}/cache/b9/{pre}_{t}_seed{seed}.pkl")}
        m = C["STm"]; eps = sorted(m["test"]); out = {}
        ta = {e: ap_idx(*fmatch(gt, m["test"][e]), np.arange(nt)) for e in eps}; ta[0] = base
        fmv = {e: fmatch(gv, m["val"][e])[0] for e in eps}; fmv[0] = fm0v
        out["final"] = ta[eps[-1]]
        for name, sel in (("ens9", [e for e in eps if e >= 10 and e % 5 == 0]), ("ens10", [e for e in eps if e > eps[-1] - 10])):
            fused = [wbf([m["test"][e][i] for e in sel], len(sel)) for i in range(nt)]
            out[name] = ap_idx(*fmatch(gt, fused), np.arange(nt))
        D = draws(sid, seed, 10, "random", nv)
        out["guard10"] = float(np.mean([ta[guard(fmv, ngtv, eps, i, True)[0]] for i in D]))
        cnt = np.stack([counts(bp["val"])] + [counts(m["val"][e]) for e in eps]); dis = cnt.std(0)
        for n in (3, 5, 10):
            idx = np.argsort(-dis, kind="stable")[:n]
            out[f"dis{n}"] = ta[guard(fmv, ngtv, eps, idx, True)[0]]
            out[f"rand{n}"] = float(np.mean([ta[guard(fmv, ngtv, eps, i, True)[0]] for i in draws(sid, seed, n, "random", nv)]))
        # multi-trajectory guard: candidates keyed (pre, e); theta0 = ("0", 0)
        TA = {("0", 0): base}; FV = {("0", 0): fm0v}
        for pre, c in C.items():
            for e in sorted(c["test"]):
                TA[(pre, e)] = ta[e] if pre == "STm" else ap_idx(*fmatch(gt, c["test"][e]), np.arange(nt)); FV[(pre, e)] = fmatch(gv, c["val"][e])[0]
        order = {"0": 0, "STm": 1, "STne": 2, "STpe": 3, "STsrc": 4}; keys = sorted(TA, key=lambda k: (order[k[0]], k[1]))
        mv = []
        for i in D:
            if ngtv[i].sum() == 0: mv.append(base); continue
            sc = {k: ap_idx(FV[k], ngtv, i) for k in keys}; best = max(keys, key=lambda k: (sc[k], -order[k[0]], -k[1])); mv.append(TA[best])
        out["multi10"] = float(np.mean(mv))
        for d in (2, 5):
            vals = []
            for i in D:
                ch, sc = guard(fmv, ngtv, eps, i, False)
                if sc is None: vals.append(base); continue
                vals.append(ta[ch] if sc[ch] - ap_idx(fm0v, ngtv, i) > d else base)
            out[f"margin{d}"] = float(np.mean(vals))
        R["seeds"].append(out)
    print(t, round(base, 1), {k: round(float(np.mean([s[k] for s in R["seeds"]])), 1) for k in R["seeds"][0]}, flush=True)
    return t, R
if __name__ == "__main__":
    with Pool(8) as p: res = dict(p.map(one, SC))
    json.dump(res, open(f"{J}/results/b11_methods.json", "w"), indent=1, default=float)
    T = list(res); b = np.array([res[t]["base"] for t in T])
    for k in res[T[0]]["seeds"][0]:
        v = np.array([np.mean([s[k] for s in res[t]["seeds"]]) for t in T]); d = v - b
        print("%-8s mean %.1f helps %d hurts %d worst %.1f" % (k, v.mean(), (d > 2).sum(), (d < -2).sum(), d.min()))
    print("B11_METHODS_DONE")
