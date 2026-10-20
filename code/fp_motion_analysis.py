"""Are the teacher's false pseudo-labels static? For each pool frame t and each teacher box (tau=0.45), persistence =
share of the frames t-10..t+10 that hold a teacher box at the same place (IoU>=0.5). Static warm objects should be
persistent; moving people should not. Reports, per scenario, the share of true and false pseudo-boxes that are
static (persistence>=0.8) and the pseudo-label precision/recall after dropping static boxes."""
import pickle, numpy as np
from jcommon import *
import cr_eval as E
TAU, Wn, PS = 0.45, 10, 0.8
print(f"{'sid':>3} {'scenario':32s} {'boxes':>6} {'P':>5} {'R':>5} | {'TP static':>9} {'FP static':>9} | {'P after':>7} {'R after':>7}")
for sid, name, split, imgs, labs in scenarios():
    rec = pickle.load(open(f"{J}/cache/teacher_s{sid:02d}_pool.pkl", "rb"))
    rec = sorted(rec, key=lambda r: r["im"])
    D = [r["det"][r["conf"] >= TAU] for r in rec]; G = [r["gt"] for r in rec]
    tp_flags, static_flags = [], []; ngt = sum(len(g) for g in G); tp_keep = 0; kept = 0
    for t, (d, g) in enumerate(zip(D, G)):
        if len(d) == 0: continue
        used = np.zeros(len(g), bool); M = E.iou_mat(d, g); tpf = np.zeros(len(d), bool)
        for k in range(len(d)):
            if len(g) == 0: break
            row = np.where(used, -1, M[k]); j = int(np.argmax(row))
            if row[j] >= 0.5: used[j] = True; tpf[k] = True
        lo, hi = max(0, t - Wn), min(len(D), t + Wn + 1); nb = [D[u] for u in range(lo, hi) if u != t]
        pers = np.array([np.mean([(len(x) > 0 and E.iou_mat(d[k:k+1], x).max() >= 0.5) for x in nb]) if nb else 0 for k in range(len(d))])
        st = pers >= PS
        tp_flags += list(tpf); static_flags += list(st)
        kept += int((~st).sum()); tp_keep += int((tpf & ~st).sum())
    tp_flags = np.array(tp_flags, bool); static_flags = np.array(static_flags, bool); nb_ = len(tp_flags)
    P = tp_flags.mean() if nb_ else 0; R = tp_flags.sum() / max(ngt, 1)
    tps = static_flags[tp_flags].mean() if tp_flags.any() else float("nan"); fps = static_flags[~tp_flags].mean() if (~tp_flags).any() else float("nan")
    print(f"{sid:>3} {name[:32]:32s} {nb_:6d} {100*P:5.1f} {100*R:5.1f} | {100*tps:9.1f} {100*fps:9.1f} | {100*tp_keep/max(kept,1):7.1f} {100*tp_keep/max(ngt,1):7.1f}", flush=True)
