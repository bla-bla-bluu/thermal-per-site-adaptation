"""Batch 11d: weight-space interpolation between the source detector and the adapted student, a label-free
safeguard that costs nothing at inference (one detector is deployed; cf. WiSE-FT, Wortsman et al. 2022). theta(alpha) =
(1 - alpha) * theta_0 + alpha * theta_student for every floating-point weight and buffer (BatchNorm statistics included; integer
buffers from the student), alpha in {0.25, 0.5, 0.75, 1}; alpha = 1 is the final-epoch student and must reproduce the cached
final-epoch AP50. The student is the last.pt of the main configuration (EMA weights), seeds 0 to 2. Settings in which the source
detector and the student share the architecture and the class head: the 23 recordings (LLVIP YOLOv8-Nano), YOLOv8-Small, CAMEL and
BIRDSAI, Cityscapes -> Foggy Cityscapes (the COCO source of MOT and of the 'coco' row has an 80-class head: not applicable).
Evaluation: the paper's pipeline (640 px, conf >= 0.001, NMS 0.5, 300 boxes; Cityscapes 1280 px). Writes results/b11_wise.json."""
import os, sys, json, glob, pickle
import numpy as np
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
from b9_policy import fmatch, ap_idx, L
from selection_study_helpers import load_gt
from PIL import Image
from ultralytics import YOLO
import torch
DEV = "cuda:0"; ALPHAS = (0.25, 0.5, 0.75, 1.0)
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def arrays(ps): return [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
def predict(model, arrs, imgsz):
    out = []
    for i in range(0, len(arrs), 16):
        for r in model.predict(arrs[i:i + 16], imgsz=imgsz, conf=0.001, iou=0.5, max_det=300, verbose=False, device=DEV):
            out.append((r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy().reshape(-1)))
    return out
def sd_of(path): return {k: v.detach().cpu() for k, v in YOLO(path).model.float().state_dict().items()}
def blend(sd0, sd1, a):
    return {k: ((1 - a) * sd0[k].float() + a * v.float()).to(v.dtype) if v.dtype.is_floating_point else v.clone() for k, v in sd1.items()}
def evaluate(src_path, stud_paths, imgs, imgsz, label):
    gt = load_gt(imgs); arrs = arrays(imgs); sd0 = sd_of(src_path); res = {a: [] for a in ALPHAS}
    base = ap_idx(*fmatch(gt, predict(YOLO(src_path), arrs, imgsz)), np.arange(len(gt))); res["base"] = base
    for sp in stud_paths:
        if not os.path.exists(sp): continue
        sd1 = sd_of(sp)
        for a in ALPHAS:
            m = YOLO(sp); m.model.float(); m.model.load_state_dict(blend(sd0, sd1, a)); m.model.to(DEV)      # fresh model: predict() fuses Conv+BN in place
            res[a].append(ap_idx(*fmatch(gt, predict(m, arrs, imgsz)), np.arange(len(gt))))
    print(label, round(base, 1), {a: round(float(np.mean(res[a])), 1) for a in ALPHAS if res[a]}, flush=True)
    return {str(k): v for k, v in res.items()}
def flc_job(sid):
    t = tag(sid); imgs = sorted(glob.glob(f"{sdir(sid, 'test')}/images/*.jpg"))
    return t, evaluate(BASELINE, [f"{J}/runs/STm_{t}_t055_seed{k}/weights/last.pt" for k in (0, 1, 2)], imgs, 640, f"flc {t}")
if __name__ == "__main__":
    # the interpolated detector collapses (AP50 0 at alpha = 0.5), so only the 23 recordings are evaluated, four at a time
    from multiprocessing import get_context
    SC = [s for s, *_ in scenarios()]
    with get_context("spawn").Pool(4) as pool: flc = dict(pool.map(flc_job, SC))
    json.dump({"flc": flc}, open(f"{J}/results/b11_wise.json", "w"), indent=1, default=float)
    b = np.array([x["base"] for x in flc.values()]); s = f"23 recordings base {b.mean():5.1f}"
    for a in ALPHAS:
        v = np.array([np.mean(x[str(a)]) for x in flc.values()]); dd = v - b; s += " | a=%.2f %5.1f (%d/%d, worst %.1f)" % (a, v.mean(), (dd > 2).sum(), (dd < -2).sum(), dd.min())
    print(s); print("B11D_DONE")
