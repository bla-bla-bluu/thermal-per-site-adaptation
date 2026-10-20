"""Label-free recording-level signals from background differences of the teacher boxes (tau 0.55, all pool frames):
median fraction, and share of boxes below 0.3 / 0.4 / 0.5. Writes results/b11_bgsignal.json."""
import os, pickle, json, numpy as np
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
out = {}
for sid, *_ in scenarios():
    t = f"s{sid:02d}" if sid < 100 else f"s{sid}"; R = pickle.load(open(f"{J}/cache/b11/fg_{t}.pkl", "rb"))
    f = np.concatenate([r["fg"][r["conf"] >= 0.55] for r in R]) if R else np.zeros(0)
    out[t] = {"n": int(len(f)), "median": float(np.median(f)) if len(f) else 0.0,
              **{f"share_lt{int(x*10)}": float((f < x).mean()) if len(f) else 0.0 for x in (0.3, 0.4, 0.5)}}
json.dump(out, open(f"{J}/results/b11_bgsignal.json", "w"), indent=1); print(json.dumps(out))
