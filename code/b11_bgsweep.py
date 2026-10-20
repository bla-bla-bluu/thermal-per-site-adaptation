"""Gate threshold f (keep a teacher box if its background-difference fraction >= f), chosen without the target's labels:
leave one scene out, f maximises the pseudo-label F1 summed over the labelled pool frames of the other scenes'
recordings (the target scene's labels are never used). Prints per-recording P/R at each f and the LOSO choice."""
import os, pickle, glob, json, numpy as np
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
FS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SCENE = {"s03": "s03", "s20": "s03", "s10": "s10", "s11": "s10", "s18": "s18", "s19": "s18"}   # same place = same scene
def counts(out, labelled, f):
    tp = nb = ng = 0
    for r in out:
        if os.path.splitext(r["im"])[0] not in labelled: continue
        sel = (r["conf"] >= 0.55) & (r["fg"] >= f); D = r["det"][sel]; G = r["gt"]; nb += len(D); ng += len(G)
        if len(D) and len(G):
            M = E.iou_mat(D, G); used = np.zeros(len(G), bool)
            for k in range(len(D)):
                row = np.where(used, -1, M[k]); j = int(np.argmax(row))
                if row[j] >= 0.5: used[j] = True; tp += 1
    return tp, nb, ng
C = {}
for sid, *_ in scenarios():
    t = f"s{sid:02d}" if sid < 100 else f"s{sid}"; out = pickle.load(open(f"{J}/cache/b11/fg_{t}.pkl", "rb"))
    labelled = {x[:-4] for x in os.listdir(f"{sdir(sid, 'pool')}/labels") if x.endswith(".txt")}
    C[t] = {f: counts(out, labelled, f) for f in FS}
def f1(tp, nb, ng): p = tp / max(nb, 1); r = tp / max(ng, 1); return 2 * p * r / max(p + r, 1e-9)
res = {}
for t in C:
    sc = SCENE.get(t, t); others = [u for u in C if SCENE.get(u, u) != sc]
    best = max(FS, key=lambda f: (np.mean([f1(*C[u][f]) for u in others]), -f))
    tp, nb, ng = C[t][best]; tp0, nb0, ng0 = C[t][0.0]
    res[t] = {"f": best, "P0": tp0 / max(nb0, 1), "R0": tp0 / max(ng0, 1), "P": tp / max(nb, 1), "R": tp / max(ng, 1)}
    print(t, "LOSO f", best, "P %.2f->%.2f R %.2f->%.2f" % (res[t]["P0"], res[t]["P"], res[t]["R0"], res[t]["R"]),
          "| F1 by f:", " ".join("%.2f" % f1(*C[t][f]) for f in FS))
json.dump(res, open(f"{J}/results/b11_bgsweep.json", "w"), indent=1)
