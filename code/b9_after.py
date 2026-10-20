"""Batch 9e (journal revision; after b9_train.py):
  unify  test AP50 of the threshold-sweep students of seeds 1 and 2 (unified pipeline) -> results/unified_test_ap.json
  sed    self-entropy descent (SED, Li et al., AAAI 2021) as a label-free threshold choice for a one-class detector: the
         mean binary entropy of each final-epoch student's detections (conf >= 0.05) on the pool frames, for
         tau = 0.65, 0.55, 0.45, 0.35, 0.25 (tau = 0.55: the main students STm); SED lowers the threshold from 0.65 while
         the entropy decreases and keeps the threshold at the first local minimum
  cross  cross-session test: final-epoch main students (STm, three seeds) of one recording on the test window of the
         other recording of the same place (s03/s20, s10/s11, s18/s19)
Writes results/b9_after.json."""
import os, glob, json, pickle
import numpy as np
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
from selection_study_helpers import load_gt, predict
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def frames(sid, part): return sorted(glob.glob(f"{sdir(sid, part)}/images/*.jpg"))
def arrays(ps): return [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
def summ(gt, det):
    d, g = E.match([{"gt": a, "det": b, "conf": c, "gain": 1.0} for a, (b, c) in zip(gt, det)]); m = E.summarize(d, len(g))
    return {"AP50": 100 * m["AP50"], "R": 100 * m["R"], "P": 100 * m["P"], "F1": 100 * m["F1"]}
RUN = {"0.25": "ST_{t}_t025_seed{k}", "0.35": "ST_{t}_t035_seed{k}", "0.45": "ST_{t}_t045_seed{k}", "0.55": "STm_{t}_t055_seed{k}",
       "0.65": "ST_{t}_t065_seed{k}"}
SC = [s for s, *_ in scenarios()]; UF = f"{J}/results/unified_test_ap.json"; U = json.load(open(UF)); res = {"sed": {}, "cross": {}}
for sid in SC:
    t = tag(sid); pa = arrays(frames(sid, "pool")); tp = frames(sid, "test"); ta, gt = arrays(tp), load_gt(tp); r = res["sed"][t] = {}
    for tau, pat in RUN.items():
        for k in (0, 1, 2):
            w = f"{J}/runs/{pat.format(t=t, k=k)}/weights/last.pt"
            if not os.path.exists(w): print("missing", w, flush=True); continue
            m = YOLO(w); c = np.concatenate([cc for _, cc in predict(m, pa)]); c = c[c >= 0.05]
            se = float(np.mean(-(c * np.log(c) + (1 - c) * np.log(np.clip(1 - c, 1e-12, 1))))) if len(c) else 0.0
            s = summ(gt, predict(m, ta)); r[f"{tau}|{k}"] = {"se": se, "n": int(len(c)), "AP50": s["AP50"]}
            if tau in ("0.25", "0.35", "0.65") and k: U.setdefault(t, {})[f"st_t{tau.replace('.', '')}_seed{k}"] = s
    print("sed", t, {kk: round(v["se"], 3) for kk, v in r.items()}, flush=True)
json.dump(U, open(UF, "w"), indent=1)
for a, b in ((3, 20), (20, 3), (10, 11), (11, 10), (18, 19), (19, 18)):
    tp = frames(b, "test"); ta, gt = arrays(tp), load_gt(tp)
    res["cross"][f"{tag(a)}->{tag(b)}"] = {k: summ(gt, predict(YOLO(f"{J}/runs/STm_{tag(a)}_t055_seed{k}/weights/last.pt"), ta))["AP50"] for k in (0, 1, 2)}
    print("cross", a, b, res["cross"][f"{tag(a)}->{tag(b)}"], flush=True)
json.dump(res, open(f"{J}/results/b9_after.json", "w"), indent=1)
print("B9_AFTER_DONE")
