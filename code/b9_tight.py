"""Box tightness: mean IoU of true-positive matches (IoU >= 0.5) with the ground truth, for the teacher pseudo-labels and
the source detector's own boxes on the labelled pool frames (conf >= 0.55), and for theta0 and the final students on the
test frames (conf >= 0.25). Writes results/b9_tight.json."""
import os, json, pickle, sys
import numpy as np
os.environ["JDOMAIN"] = "all"; sys.path.insert(0, "/path/to/domain/journal/code")
from jcommon import *
import cr_eval as E
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
def tp_ious(G, D):
    out = []
    for g, d in zip(G, D):
        if len(g) == 0 or len(d) == 0: continue
        M = E.iou_mat(d, g); used = np.zeros(len(g), bool)
        for k in range(len(d)):
            row = np.where(used, -1, M[k]); j = int(np.argmax(row))
            if row[j] >= 0.5: used[j] = True; out.append(float(row[j]))
    return out
srt = lambda d, c, th: d[np.argsort(-c, kind="stable")][np.sort(c)[::-1] >= th]
res = {"teacher_pool": [], "theta0_pool": [], "theta0_test": [], "final_test": []}
for sid, *_ in scenarios():
    t = tag(sid); rec = sorted(pickle.load(open(f"{J}/cache/teacher_{t}_pool.pkl", "rb")), key=lambda r: r["im"])
    lab = {f[:-4] for f in os.listdir(f"{sdir(sid, 'pool')}/labels") if f.endswith(".txt")}
    bp = pickle.load(open(f"{J}/cache/Baseline_{t}_pool_predict.pkl", "rb"))
    keep = [i for i, r in enumerate(rec) if os.path.splitext(r["im"])[0] in lab]
    res["teacher_pool"] += tp_ious([rec[i]["gt"] for i in keep], [srt(rec[i]["det"], rec[i]["conf"], 0.55) for i in keep])
    res["theta0_pool"] += tp_ious([rec[i]["gt"] for i in keep], [srt(*bp[i], 0.55) for i in keep])
    gt = pickle.load(open(f"{J}/cache/b9/gt_{t}_test.pkl", "rb")); b6 = pickle.load(open(f"{J}/cache/b6/baseline_{t}.pkl", "rb"))["test"]
    res["theta0_test"] += tp_ious(gt, [srt(d, c, 0.25) for d, c in b6])
    for k in range(3):
        st = pickle.load(open(f"{J}/cache/b9/STm_{t}_seed{k}.pkl", "rb"))["test"][50]
        res["final_test"] += tp_ious(gt, [srt(d, c, 0.25) for d, c in st])
out = {k: {"n": len(v), "mean_iou": float(np.mean(v)), "share_075": float(np.mean(np.array(v) >= 0.75))} for k, v in res.items()}
json.dump(out, open(f"{J}/results/b9_tight.json", "w"), indent=1); print(out)
