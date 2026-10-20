"""Quick look: final-epoch test AP50 (mean over available seeds) of the batch-9 variants, from the per-epoch caches."""
import os, glob, pickle, sys
import numpy as np
os.environ["JDOMAIN"] = "all"; sys.path.insert(0, "/path/to/domain/journal/code")
from jcommon import *
import cr_eval as E
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
def ap(gt, det):
    d, g = E.match([{"gt": a, "det": b, "conf": c, "gain": 1.0} for a, (b, c) in zip(gt, det)]); return 100 * E.summarize(d, len(g))["AP50"]
rows = []
for sid, *_ in scenarios():
    t = tag(sid); gt = pickle.load(open(f"{J}/cache/b9/gt_{t}_test.pkl", "rb")); b = ap(gt, pickle.load(open(f"{J}/cache/b6/baseline_{t}.pkl", "rb"))["test"])
    r = [t, b]
    for pre in ("STm", "STne", "STns"):
        v = [ap(gt, pickle.load(open(f, "rb"))["test"][50]) for f in sorted(glob.glob(f"{J}/cache/b9/{pre}_{t}_seed*.pkl"))]
        r.append((np.mean(v), len(v)) if v else (np.nan, 0))
    rows.append(r); print(t, f"base {b:5.1f}", " ".join(f"{p} {m:5.1f}({n})" for p, (m, n) in zip(("STm", "STne", "STns"), r[2:])), flush=True)
for i, p in enumerate(("STm", "STne", "STns")):
    v = [r[2 + i][0] for r in rows if r[2 + i][1] == 3]; print(p, "mean over complete recordings", round(np.mean(v), 1), len(v))
