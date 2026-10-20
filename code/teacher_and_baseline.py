"""(1) Teacher = urban baseline + SAHI (320-px tiles, 0.2 overlap, NMS IoU 0.5, conf floor 0.001) on pool and gap frames:
raw detections cached, so any pseudo-label threshold can be applied later without rerunning SAHI.
(2) Pseudo-label sets at tau = 0.45 (the ICVGIP value, not re-tuned) + pseudo-label quality vs the hidden pool labels.
(3) Baseline evaluation on the test split: standard inference and SAHI."""
import os, json, pickle
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
os.makedirs(f"{J}/cache", exist_ok=True); os.makedirs(f"{J}/results", exist_ok=True)
def cached(key, fn):
    p = f"{J}/cache/{key}.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    v = fn(); pickle.dump(v, open(p, "wb")); return v
TAUS = (0.25, 0.35, 0.45, 0.55)
out = open(f"{J}/results/teacher_baseline.jsonl", "a")
for sid, name, split, imgs, labs in scenarios():
    for part in ("pool", "gap"):
        rec = cached(f"teacher_s{sid:02d}_{part}", lambda: E.gen_sahi(BASELINE, 320, img_dir=f"{sdir(sid, part)}/images"))
        if part != "pool":
            continue
        q = {}
        labelled = {f[:-4] for f in os.listdir(f"{sdir(sid, 'pool')}/labels") if f.endswith(".txt")}
        for tau in TAUS:          # pseudo-label quality at several thresholds (labels of the pool are hidden from training)
            tp = fp = ng = nb = 0
            for r in rec:
                if os.path.splitext(r["im"])[0] not in labelled:   # forest pools: unlabeled video frames are skipped
                    continue
                keep = r["conf"] >= tau; D = r["det"][keep]; G = r["gt"]; nb += len(D); ng += len(G)
                M = E.iou_mat(D, G); used = np.zeros(len(G), bool)
                for k in np.argsort(-r["conf"][keep]):
                    if len(G) == 0: break
                    row = np.where(used, -1, M[k]); j = int(np.argmax(row))
                    if row[j] >= 0.5: used[j] = True; tp += 1
                fp += len(D) - int(used.sum())
            q[str(tau)] = {"boxes_per_frame": nb / max(len(labelled), 1), "P": tp / max(nb, 1), "R": tp / max(ng, 1)}
        # write the tau = 0.45 pseudo-label set
        d = f"{J}/data/flc/s{sid:02d}/pl_t045"; os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
        for r in rec:
            stem = os.path.splitext(r["im"])[0]; src = f"{sdir(sid, 'pool')}/images/{r['im']}"
            if not os.path.islink(f"{d}/images/{r['im']}"):
                os.symlink(os.path.realpath(src), f"{d}/images/{r['im']}")
            W, H = map(float, Image.open(src).size)
            keep = r["conf"] >= 0.45
            open(f"{d}/labels/{stem}.txt", "w").write("".join(
                f"0 {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n" for b in r["det"][keep]))
        open(f"{J}/data/flc/s{sid:02d}_st045.yaml", "w").write(
            f"path: {d}\ntrain: images\nval: {sdir(sid, 'val')}/images\nnames:\n  0: person\n")
        open(f"{J}/data/flc/s{sid:02d}_sup.yaml", "w").write(
            f"path: {sdir(sid, 'pool_lab') if os.path.isdir(sdir(sid, 'pool_lab')) else sdir(sid, 'pool')}\ntrain: images\nval: {sdir(sid, 'val')}/images\nnames:\n  0: person\n")
        out.write(json.dumps({"sid": sid, "name": name, "kind": "pseudo_label_quality", "q": q}) + "\n"); out.flush()
    for mode in ("std", "sahi"):
        if mode == "std":
            rec, _ = cached(f"Baseline_s{sid:02d}_test_std", lambda: E.gen_std(BASELINE, f"{J}/data/flc/s{sid:02d}_test.yaml"))
        else:
            rec = cached(f"Baseline_s{sid:02d}_test_sahi", lambda: E.gen_sahi(BASELINE, 320, img_dir=f"{sdir(sid, 'test')}/images"))
        dets, gts = E.match(rec); m = E.summarize(dets, len(gts)); m["CI"] = E.bootstrap(dets, gts, len(rec), n=2000); m["frames"] = len(rec)
        out.write(json.dumps({"sid": sid, "name": name, "model": "Baseline", "mode": mode, **m}) + "\n"); out.flush()
        print(sid, name, "Baseline", mode, f"AP50 {100*m['AP50']:.1f} R {100*m['R']:.1f} P {100*m['P']:.1f}", flush=True)
print("TEACHER_BASELINE_DONE")
