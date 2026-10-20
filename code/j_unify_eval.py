"""Journal: re-score every final checkpoint with the frame-by-frame YOLO.predict pipeline used for the per-epoch
studies (B6), so that all numbers in the paper come from one pipeline. Output: results/unified_test_ap.json
{scene: {model: AP50 %}}; also R/P/F1 at conf 0.15 for the baseline and the main students."""
import os, json, glob, pickle
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
from selection_study_helpers import load_gt, predict
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
OUT = f"{J}/results/unified_test_ap.json"; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for sid, *_ in scenarios():
    t = tag(sid); ps = sorted(glob.glob(f"{sdir(sid, 'test')}/images/*.jpg")); gt = load_gt(ps)
    arr = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
    models = {"baseline": BASELINE}
    for k in (0, 1, 2):
        models[f"st045_seed{k}"] = f"{J}/runs/ST_{t}_t045_seed{k}/weights/last.pt" if k else f"{J}/runs/ST_{t}_t045_seed0/weights/last.pt"
        models[f"sup_seed{k}"] = f"{J}/runs/SUP_{t}_seed{k}/weights/last.pt"
    for v in ("t025", "t035", "t055", "t065", "pers"):
        models[f"st_{v}_seed0"] = f"{J}/runs/ST_{t}_{v}_seed0/weights/last.pt"
    models["stsrc_seed0"] = f"{J}/runs/STsrc_{t}_t045_seed0/weights/last.pt"
    models["st045_valbest_seed0"] = f"{J}/runs/ST_{t}_t045_seed0/weights/best.pt"
    r = res.setdefault(t, {})
    for name, w in models.items():
        if name in r or not os.path.exists(w): continue
        det = predict(YOLO(w), arr)
        rec = [{"gt": g, "det": d, "conf": c, "gain": 1.0} for g, (d, c) in zip(gt, det)]
        dd, gg = E.match(rec); m = E.summarize(dd, len(gg))
        r[name] = {"AP50": 100 * m["AP50"], "R": 100 * m["R"], "P": 100 * m["P"], "F1": 100 * m["F1"]}
    json.dump(res, open(OUT, "w"), indent=1); print("scored", t, len(r), flush=True)
print("UNIFY_DONE")
