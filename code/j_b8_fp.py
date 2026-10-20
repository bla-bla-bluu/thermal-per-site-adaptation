"""False pseudo-labels by type, on the labelled pool frames, at tau 0.45 and 0.55: a teacher box is correct if it matches
a person at IoU >= 0.5 (greedy by score); a false one overlaps a person (IoU > 0.1: a fragment or a duplicate), else an
animal (FLC-Rural labels, class 1), else it lies on the background. Writes results/fp_types_tau.json {tau: {sid: counts}}."""
import os, json, pickle
import numpy as np
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
ORIG = {s_: labs for s_, _, _, _, labs in scenarios()}   # FLC split labels are person-only; animals come from the release
def boxes(path, W, H, cls):
    out = []
    for l in open(path):
        q = l.split()
        if len(q) == 5 and q[0] == cls:
            cx, cy, w, h = map(float, q[1:]); out.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
    return np.array(out).reshape(-1, 4)
res = {}
for s in ORIG:
    rec = pickle.load(open(f"{J}/cache/teacher_{tag(s)}_pool.pkl", "rb")); frames = []
    for r in rec:
        img = f"{sdir(s, 'pool')}/images/{r['im']}"; lab = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lab): continue
        W, H = Image.open(img).size; g = boxes(lab, W, H, "0")
        a = boxes(ORIG[s][int(r["im"].split(".")[0])], W, H, "1") if s < 100 else np.zeros((0, 4))
        frames.append((r, g, a))
    for tau in (0.45, 0.55):
        c = dict(tp=0, fp_person=0, fp_animal=0, fp_background=0, n_gt=0)
        for r, g, a in frames:
            c["n_gt"] += len(g); keep = r["conf"] >= tau; d = r["det"][keep][np.argsort(-r["conf"][keep])]; used = np.zeros(len(g), bool)
            for b in d:
                iou = E.iou_mat(b[None], g)[0] if len(g) else np.zeros(0)
                if len(g):
                    row = np.where(used, -1, iou); j = int(np.argmax(row))
                    if row[j] >= 0.5: used[j] = True; c["tp"] += 1; continue
                if len(g) and iou.max() > 0.1: c["fp_person"] += 1
                elif len(a) and E.iou_mat(b[None], a)[0].max() > 0.1: c["fp_animal"] += 1
                else: c["fp_background"] += 1
        n = c["tp"] + c["fp_person"] + c["fp_animal"] + c["fp_background"]
        c.update(frames=len(frames), precision=c["tp"] / max(n, 1), recall=c["tp"] / max(c["n_gt"], 1))
        res.setdefault(str(tau), {})[s] = c
    print(s, {t: (res[t][s]["tp"], res[t][s]["fp_person"], res[t][s]["fp_animal"], res[t][s]["fp_background"]) for t in res}, flush=True)
json.dump(res, open(f"{J}/results/fp_types_tau.json", "w"), indent=1); print("FP_DONE")
