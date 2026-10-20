"""Batch 12: the paper's pipeline with other source detectors, on the 23 recordings.
  v8s    YOLOv8-Small trained on LLVIP like theta0 (the ICVGIP camera-ready baseline runs/CR_Base_v8s, 50 epochs, 640 px);
         students YOLOv8-Small from COCO
  coco   the COCO-pretrained YOLOv8-Nano itself, person class only (a visible-light source dataset); students YOLOv8-Nano
         from COCO
Everything else as in the paper: teacher = source + SAHI (320-px tiles, overlap 0.2, full frame, NMS 0.5), tau = 0.55,
50 epochs at 640 px, batch 16, seeds 0-2, every epoch's predictions on the validation and test windows cached (then the
epoch weights are deleted), ten-frame guard with the source model as a candidate (same draws as the paper).
Steps: prep <src> | train <npar> | epochs <worker> <nworkers> | policy <src>. Writes results/b12_<src>.json."""
import os, sys, glob, json, pickle, time, subprocess
import numpy as np
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
SC = [s for s, *_ in scenarios()]; PY = "/opt/conda/bin/python"; TAU = 0.55
SRC = {"v8s": dict(w="/path/to/domain/runs/CR_Base_v8s/weights/best.pt", init="yolov8s.pt", cls=None),
       "v8m": dict(w=f"{J}/runs/J_Base_v8m/weights/best.pt", init=f"{J}/yolov8m.pt", cls=None),
       "coco": dict(w="yolov8n.pt", init="yolov8n.pt", cls=[0])}
CD = f"{J}/cache/b12"; os.makedirs(CD, exist_ok=True)
SRCS = os.environ.get("B12_SRCS", "v8s,coco").split(",")      # B12_SRCS=v8m runs only the YOLOv8-Medium source (batch 12b)
def sahi(weights, img_dir, cls):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=weights, confidence_threshold=0.001, device="cuda:0")
    rec = []
    for p in sorted(glob.glob(f"{img_dir}/*.jpg")):
        r = get_sliced_prediction(p, dm, slice_height=320, slice_width=320, overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                  postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5, verbose=0)
        ob = [o for o in r.object_prediction_list if cls is None or o.category.id in cls]
        rec.append({"im": os.path.basename(p), "det": np.array([[o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy] for o in ob]).reshape(-1, 4),
                    "conf": np.array([o.score.value for o in ob])})
    return rec
def predict(model, paths, cls=None):
    out = []
    for i in range(0, len(paths), 32):
        for r in model.predict(paths[i:i + 32], imgsz=640, conf=0.001, iou=0.5, max_det=300, verbose=False, device="cuda:0", classes=cls):
            out.append((r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy().reshape(-1)))
    return out
def prep(src):
    from ultralytics import YOLO
    S = SRC[src]; m = YOLO(S["w"])
    for sid in SC:
        t = tag(sid); tf = f"{CD}/{src}_teacher_{t}.pkl"
        if not os.path.exists(tf): pickle.dump(sahi(S["w"], f"{sdir(sid, 'pool')}/images", S["cls"]), open(tf, "wb"))
        rec = pickle.load(open(tf, "rb")); d = f"{J}/data/flc/{t}/pl_{src}055"
        os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
        for r in rec:
            src_img = os.path.realpath(f"{sdir(sid, 'pool')}/images/{r['im']}")
            if not os.path.islink(f"{d}/images/{r['im']}"): os.symlink(src_img, f"{d}/images/{r['im']}")
            W, H = map(float, Image.open(src_img).size); k = r["conf"] >= TAU
            open(f"{d}/labels/{os.path.splitext(r['im'])[0]}.txt", "w").write("".join(
                f"0 {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n" for b in r["det"][k]))
        open(f"{J}/data/flc/{t}_{src}055.yaml", "w").write(f"path: {d}\ntrain: images\nval: {sdir(sid, 'val')}/images\nnames:\n  0: person\n")
        bf = f"{CD}/{src}_base_{t}.pkl"
        if not os.path.exists(bf):
            pickle.dump({p: predict(m, sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg")), S["cls"]) for p in ("val", "test")}, open(bf, "wb"))
        print(time.strftime("%H:%M:%S"), "prep", src, t, "boxes/frame", round(np.mean([(r["conf"] >= TAU).sum() for r in rec]), 1), flush=True)
    print("B12_PREP_DONE", src, flush=True)
def jobs():
    return [(src, sid, seed) for src in SRCS for seed in (0, 1, 2) for sid in SC]
def rname(src, sid, seed): return f"B12{src}_{tag(sid)}_seed{seed}"
def train(npar):
    running = []
    for src, sid, seed in jobs():
        n = rname(src, sid, seed)
        if os.path.exists(f"{J}/runs/{n}/DONE") or not os.path.exists(f"{J}/data/flc/{tag(sid)}_{src}055.yaml"): continue
        while len([p for p in running if p.poll() is None]) >= npar: time.sleep(10)
        code = (f"from ultralytics import YOLO; YOLO('{SRC[src]['init']}').train(data='{J}/data/flc/{tag(sid)}_{src}055.yaml', epochs=50, imgsz=640, "
                f"batch=16, device='cuda:0', workers=4, cache='ram', project='{J}/runs', name='{n}', exist_ok=True, patience=50, seed={seed}, "
                f"plots=False, save_period=1); open('{J}/runs/{n}/DONE','w').write('ok')")
        running.append(subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{n}.log", "w"), stderr=subprocess.STDOUT, cwd=J))
        print(time.strftime("%H:%M:%S"), "launched", n, flush=True)
    for p in running: p.wait()
    open(f"{J}/runs/B12_TRAIN_DONE", "w").write("ok"); print("B12_TRAIN_DONE", flush=True)
def epochs(wk, nw):
    from ultralytics import YOLO
    os.makedirs(f"{CD}/locks", exist_ok=True)
    while True:
        done_train = os.path.exists(f"{J}/runs/B12_TRAIN_DONE"); pending = 0
        for i, (src, sid, seed) in enumerate(jobs()):
            n = rname(src, sid, seed); cf = f"{CD}/{n}.pkl"
            if os.path.exists(cf): continue
            if not os.path.exists(f"{J}/runs/{n}/DONE"): pending += 1; continue
            try: os.mkdir(f"{CD}/locks/{n}")
            except FileExistsError: continue
            P = {p: sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg")) for p in ("val", "test")}; out = {"val": {}, "test": {}}
            cks = sorted(glob.glob(f"{J}/runs/{n}/weights/epoch*.pt"), key=lambda q: int(os.path.basename(q)[5:-3]))
            for ck in cks:
                e = int(os.path.basename(ck)[5:-3]) + 1; m = YOLO(ck)
                for p in out: out[p][e] = predict(m, P[p])
            pickle.dump(out, open(cf + ".tmp", "wb")); os.replace(cf + ".tmp", cf)
            for ck in cks: os.remove(ck)
            print(time.strftime("%H:%M:%S"), "cached", n, len(cks), flush=True)
        if done_train and pending == 0: break
        time.sleep(60)
    print("B12_EPOCHS_DONE", wk, flush=True)
def policy(src):
    from b9_policy import fmatch, ap_idx, guard, draws, L
    res = {}
    for sid in SC:
        t = tag(sid); gv, gt = L(f"{J}/cache/b9/gt_{t}_val.pkl"), L(f"{J}/cache/b9/gt_{t}_test.pkl"); nv, nt = len(gv), len(gt)
        b = L(f"{CD}/{src}_base_{t}.pkl"); fm0v, ngtv = fmatch(gv, b["val"]); base = ap_idx(*fmatch(gt, b["test"]), np.arange(nt))
        V = {"base": base, "final": [], "pick10": [], "pick10+base": [], "opt": []}
        for seed in (0, 1, 2):
            cf = f"{CD}/{rname(src, sid, seed)}.pkl"
            if not os.path.exists(cf): continue
            c = L(cf); eps = sorted(c["test"]); ta = {e: ap_idx(*fmatch(gt, c["test"][e]), np.arange(nt)) for e in eps}; ta[0] = base
            fmv = {e: fmatch(gv, c["val"][e])[0] for e in eps}; fmv[0] = fm0v
            V["final"].append(ta[eps[-1]]); V["opt"].append(max(ta.values()))
            for wb in (False, True):
                V["pick10+base" if wb else "pick10"].append(float(np.mean([ta[guard(fmv, ngtv, eps, i, wb)[0]] for i in draws(sid, seed, 10, "random", nv)])))
        res[t] = V; print("policy", src, t, round(base, 1), {k: round(float(np.mean(v)), 1) for k, v in V.items() if isinstance(v, list) and v}, flush=True)
    json.dump(res, open(f"{J}/results/b12_{src}.json", "w"), indent=1, default=float); print("B12_POLICY_DONE", src)
if __name__ == "__main__":
    s = sys.argv[1]
    if s == "prep": prep(sys.argv[2])
    elif s == "train": train(int(sys.argv[2]))
    elif s == "epochs": epochs(int(sys.argv[2]), int(sys.argv[3]))
    elif s == "policy": policy(sys.argv[2])
