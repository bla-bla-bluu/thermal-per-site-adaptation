"""Batch 2a: label-free checkpoint selection study. Usage: selection_study.py <worker> <n_workers>
For every self-training run (tau 0.45, seed 0) and every saved epoch: test AP50 (analysis only) and label-free scores
on the unlabeled gap frames (between pool and val, never trained on):
  agree    AP50 of the student's detections against the teacher's tau=0.45 SAHI boxes (pseudo-validation)
  flip     AP50 of detections on the mirrored frame against those (conf>=0.25) on the original frame
  temporal share of detections (conf>=0.25) matched at IoU>=0.3 in the adjacent frame (both directions)
  count    detections per frame at conf>=0.25
All detections come from YOLO.predict (640 px, conf 0.001, NMS IoU 0.5), so the study is internally consistent;
the labelled-validation signal is Ultralytics' per-epoch val mAP50 from results.csv."""
import os, sys, json, glob, pickle, csv
import numpy as np
from PIL import Image
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
wk, nw = int(sys.argv[1]), int(sys.argv[2])

def load_gt(img_paths):
    out = []
    for p in img_paths:
        W, H = Image.open(p).size; g = []
        for l in open(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"):
            q = l.split()
            if len(q) == 5:
                cx, cy, w, h = map(float, q[1:]); g.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
        out.append(np.array(g).reshape(-1, 4))
    return out

def predict(model, arrays):
    dets = []
    for i in range(0, len(arrays), 32):
        for r in model.predict(arrays[i:i+32], imgsz=640, conf=0.001, iou=0.5, max_det=300, verbose=False, device="cuda:0"):
            dets.append((r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy().reshape(-1)))
    return dets

def ap(gt_list, det_list):
    rec = [{"gt": g, "det": d, "conf": c, "gain": 1.0} for g, (d, c) in zip(gt_list, det_list)]
    dts, gts = E.match(rec); return E.summarize(dts, len(gts))

def temporal(det_list, thr=0.25):
    hit = tot = 0
    for (d0, c0), (d1, c1) in zip(det_list[:-1], det_list[1:]):
        A, Bb = d0[c0 >= thr], d1[c1 >= thr]
        for X, Y in ((A, Bb), (Bb, A)):
            if len(X) == 0: continue
            tot += len(X)
            if len(Y): hit += int((E.iou_mat(X, Y).max(1) >= 0.3).sum())
    return hit / tot if tot else 0.0

out = open(f"{J}/results/selection_w{wk}.jsonl", "a")
for n, (sid, name, split, imgs, labs) in enumerate(scenarios()):
    if n % nw != wk: continue
    run = f"{J}/runs/ST_s{sid:02d}_t045_seed0"
    test_p = sorted(glob.glob(f"{sdir(sid, 'test')}/images/*.jpg")); gap_p = sorted(glob.glob(f"{sdir(sid, 'gap')}/images/*.jpg"))
    test_gt = load_gt(test_p)
    test_a = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in test_p]
    gap_a = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in gap_p]
    gap_f = [np.ascontiguousarray(a[:, ::-1]) for a in gap_a]; W = gap_a[0].shape[1]
    teach = pickle.load(open(f"{J}/cache/teacher_s{sid:02d}_gap.pkl", "rb"))
    tmap = {r["im"]: r["det"][r["conf"] >= 0.45] for r in teach}
    t_gt = [tmap[os.path.basename(p)] for p in gap_p]
    val = {int(r["epoch"]): float(r["metrics/mAP50(B)"]) for r in csv.DictReader(open(f"{run}/results.csv"))}
    for ep in range(50):
        ck = f"{run}/weights/epoch{ep}.pt"
        if not os.path.exists(ck): continue
        m = YOLO(ck)
        td = predict(m, test_a); gd = predict(m, gap_a); gf = predict(m, gap_f)
        gf = [(np.stack([W - d[:, 2], d[:, 1], W - d[:, 0], d[:, 3]], 1) if len(d) else d, c) for d, c in gf]
        t = ap(test_gt, td)
        rec = {"sid": sid, "epoch": ep + 1, "test_AP50": t["AP50"], "test_R": t["R"], "test_P": t["P"],
               "agree_AP50": ap(t_gt, gd)["AP50"],
               "flip_AP50": ap([d[c >= 0.25] for d, c in gd], gf)["AP50"],
               "temporal": temporal(gd), "count": float(np.mean([(c >= 0.25).sum() for _, c in gd])),
               "val_mAP50": val.get(ep + 1)}
        out.write(json.dumps(rec) + "\n"); out.flush()
    print("done", sid, name, flush=True)
print("SELECTION_DONE", wk)
