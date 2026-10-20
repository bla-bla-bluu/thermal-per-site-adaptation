"""Evaluate trained models on each scenario's test split (same matcher/AP routine as ICVGIP; bootstrap CIs)."""
import os, json, pickle
from jcommon import *
import cr_eval as E
def cached(key, fn):
    p = f"{J}/cache/{key}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    v = fn(); pickle.dump(v, open(p, "wb")); return v
out = open(f"{J}/results/flc_models.jsonl", "a")
for sid, name, split, imgs, labs in scenarios():
    jobs = [("ST-last", f"{J}/runs/ST_s{sid:02d}_t045_seed0/weights/last.pt", ("std", "sahi")),
            ("ST-valbest", f"{J}/runs/ST_s{sid:02d}_t045_seed0/weights/best.pt", ("std",)),
            ("SUP-last", f"{J}/runs/SUP_s{sid:02d}_seed0/weights/last.pt", ("std",))]
    for model, w, modes in jobs:
        if not os.path.exists(w): print("missing", w); continue
        for mode in modes:
            key = f"{model}_s{sid:02d}_test_{mode}"
            if mode == "std": rec, _ = cached(key, lambda: E.gen_std(w, f"{J}/data/flc/s{sid:02d}_test.yaml"))
            else: rec = cached(key, lambda: E.gen_sahi(w, 320, img_dir=f"{sdir(sid, 'test')}/images"))
            dets, gts = E.match(rec); m = E.summarize(dets, len(gts)); m["CI"] = E.bootstrap(dets, gts, len(rec), n=2000); m["frames"] = len(rec)
            out.write(json.dumps({"sid": sid, "name": name, "model": model, "mode": mode, **m}) + "\n"); out.flush()
            print(sid, name, model, mode, f"AP50 {100*m['AP50']:.1f} R {100*m['R']:.1f} P {100*m['P']:.1f}", flush=True)
print("EVAL_DONE")
