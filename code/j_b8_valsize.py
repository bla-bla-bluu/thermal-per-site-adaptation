"""Labelled frames needed for checkpoint selection, aggregated over the 23 recordings of the journal benchmark (20 FLC-Rural
incl. s04 = sid 103, three forest: 101, 102, 106). Same procedure as valsize_study.py (tau 0.45, seed 0 students; pick the
epoch with the best AP50 on n validation frames, random or one contiguous clip, 50 draws; look up its test AP50), run per
recording in parallel with a per-recording random seed. Writes results/valsize_groups.json."""
import os, glob, json, pickle
from multiprocessing import Pool
import numpy as np
os.environ["JDOMAIN"] = "all"
from jcommon import *
import cr_eval as E
SIDS = [s for s, *_ in scenarios()]; FOR = [101, 102, 106]; FLC = [s for s in SIDS if s not in FOR]
NS = (3, 5, 10, 20, 40, None)
test = {}
for f in glob.glob(f"{J}/results/selection_w*.jsonl"):
    for l in open(f):
        r = json.loads(l); test[(r["sid"], r["epoch"])] = 100 * r["test_AP50"]
def ap_on(gt, det, idx):
    rec = [{"gt": gt[i], "det": det[i][0], "conf": det[i][1], "gain": 1.0} for i in idx]
    d, g = E.match(rec); return E.summarize(d, len(g))["AP50"]
def one(s):
    c = pickle.load(open(f"{J}/cache/valsize/s{s:02d}.pkl", "rb")); nv = len(c["gt"]); eps = sorted(c["ep"])
    rng = np.random.default_rng(1000 + s); out = {}
    for mode in ("random", "contiguous"):
        for n in NS:
            k = nv if n is None or n >= nv else n; vals = []
            for _ in range(50 if k < nv else 1):
                if mode == "random": idx = rng.choice(nv, k, replace=False)
                else: s0 = rng.integers(0, nv - k + 1); idx = np.arange(s0, s0 + k)
                best = max(eps, key=lambda e: (ap_on(c["gt"], c["ep"][e], idx), -e)); vals.append(test[(s, best)])
            out[f"{s}|{mode}|{n}"] = float(np.mean(vals))
    print("done", s, flush=True); return out
if __name__ == "__main__":
    rows = {}
    with Pool(12) as pool:
        for r in pool.imap_unordered(one, SIDS): rows.update(r)
    out = {}
    for name, grp in (("flc", FLC), ("forest", FOR), ("all", SIDS)):
        out[name] = {"n": len(grp), "final": float(np.mean([test[(s, 50)] for s in grp]))}
        for mode in ("random", "contiguous"):
            out[name][mode] = {str(n if n else "all"): float(np.mean([rows[f"{s}|{mode}|{n}"] for s in grp])) for n in NS}
    json.dump({"groups": out, "per_scene": rows}, open(f"{J}/results/valsize_groups.json", "w"), indent=1)
    print(json.dumps(out, indent=1)); print("VALSIZE23_DONE")
