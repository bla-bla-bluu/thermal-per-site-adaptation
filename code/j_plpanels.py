"""Pseudo-label example panels for the journal draft (fixed rule, no hand picking): for each listed scene, among the
labelled pool frames, the frame with the most teacher boxes at conf >= 0.55 (ties: the earliest). Draws ground truth (green) and the
pseudo-labels, blue if matched to a ground-truth person at IoU >= 0.5 (greedy by score), orange otherwise.
Also reports the wall-clock training time of the tau=0.55 students (results.csv, 3 runs shared one GPU)."""
import os, glob, json, pickle, csv
import numpy as np
from PIL import Image, ImageDraw
import sys; sys.path.insert(0, "/path/to/domain/journal/code")
from jcommon import *
import cr_eval as E
OUT = f"{J}/results/pl_panels"; os.makedirs(OUT, exist_ok=True)
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
def gt_of(img):
    lab = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(lab): return None
    W, H = Image.open(img).size; g = []
    for l in open(lab):
        q = l.split()
        if len(q) == 5 and q[0] == "0":
            cx, cy, w, h = map(float, q[1:]); g.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
    return np.array(g).reshape(-1, 4)
meta = {}
for sid in (17, 13, 18, 103, 14, 15, 16, 3, 106):
    rec = pickle.load(open(f"{J}/cache/teacher_{tag(sid)}_pool.pkl", "rb"))
    best = None
    for r in sorted(rec, key=lambda r: r["im"]):
        img = f"{sdir(sid, 'pool')}/images/{r['im']}"; g = gt_of(img)
        if g is None: continue
        k = int((r["conf"] >= 0.55).sum())
        if best is None or k > best[0]: best = (k, img, r, g)
    k, img, r, g = best
    keep = r["conf"] >= 0.55; d, c = r["det"][keep], r["conf"][keep]
    order = np.argsort(-c); d = d[order]; used = np.zeros(len(g), bool); tp = np.zeros(len(d), bool)
    for i in range(len(d)):
        if len(g):
            iou = E.iou_mat(d[i:i+1], g)[0]; iou[used] = -1; j = int(np.argmax(iou))
            if iou[j] >= 0.5: used[j] = True; tp[i] = True
    im = Image.open(img).convert("RGB"); dr = ImageDraw.Draw(im)
    for b in g: dr.rectangle([float(v) for v in b], outline=(40, 200, 90), width=2)
    for b, t in zip(d, tp): dr.rectangle([float(v) - 2 for v in b[:2]] + [float(v) + 2 for v in b[2:]], outline=(42, 120, 214) if t else (235, 104, 52), width=2)
    im.save(f"{OUT}/{tag(sid)}.png")
    meta[tag(sid)] = {"frame": os.path.basename(img), "n_pl": int(len(d)), "tp": int(tp.sum()), "fp": int((~tp).sum()), "n_gt": int(len(g))}
    print(tag(sid), meta[tag(sid)], flush=True)
times = {}
os.environ["JDOMAIN"] = "all"
ACTIVE = {tag(s_) for s_, *_ in scenarios()}
for f in glob.glob(f"{J}/runs/STe_*_t055_seed*/results.csv"):
    if f.split("/")[-2].split("_")[1] not in ACTIVE: continue
    rows = list(csv.DictReader(open(f)))
    if rows: times[f.split("/")[-2]] = float(rows[-1]["time"])
v = np.array(list(times.values())) / 60
meta["train_minutes"] = {"n": len(v), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
print(meta["train_minutes"])
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
