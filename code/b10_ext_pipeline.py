"""Batch 10c: the paper's pipeline on the external benchmarks (data/ext/manifest.json, b10_ext_build.py).
Steps (argument): prep | train | epochs <worker> | policy
  prep    teacher (sliced inference with the YOLOv8-Nano LLVIP source detector, as in the paper) on the pool frames, cached;
          pseudo-label sets t055 (frames without pseudo-labels kept as background) and t055ne (left out); the source
          detector's val/test predictions; data yamls (also for the supervised reference on the labelled pool frames)
  train   students: EXm (t055) and EXne (t055ne), COCO YOLOv8n, 50 epochs, 640 px, batch 16, seeds 0-2, every epoch saved;
          supervised reference EXsup (labelled pool frames), seeds 0-2, final epoch
  epochs  per-epoch val/test predictions of EXm and EXne (lock per run), epoch checkpoints deleted afterwards
  policy  final epoch, ten-frame guard (with and without the source detector), oracle, for EXm and EXne; supervised;
          AP50 with the paper's matcher; writes results/b10_ext.json"""
import os, sys, glob, json, pickle, time, subprocess
import numpy as np
from PIL import Image
J = "/path/to/domain/journal"; sys.path.insert(0, f"{J}/code"); os.environ.setdefault("JDOMAIN", "all")
from jcommon import BASELINE, COCO_N
import cr_eval as E
from selection_study_helpers import load_gt, predict
MAN = json.load(open(f"{J}/data/ext/manifest.json")); SIDS = sorted(int(k) for k in MAN)
D = lambda sid: f"{J}/data/ext/{MAN[str(sid)]['dataset']}/s{sid}"; C = f"{J}/cache/ext"; RUNS = f"{J}/runs_ext"
os.makedirs(C, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
def frames(sid, part): return sorted(glob.glob(f"{D(sid)}/{part}/images/*.jpg"))
def arrays(ps): return [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
def pred_dev(model, arrs):
    """selection_study_helpers.predict on DEV (same settings: 640 px, conf 0.001, NMS IoU 0.5, at most 300 boxes)."""
    dets = []
    for i in range(0, len(arrs), 32):
        for r in model.predict(arrs[i:i + 32], imgsz=640, conf=0.001, iou=0.5, max_det=300, verbose=False, device=DEV):
            dets.append((r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy().reshape(-1)))
    return dets

def write_set(sid, var, rec, boxes, keep=None):
    d = f"{D(sid)}/pl_{var}"; os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
    for i, r in enumerate(rec):
        if keep is not None and not keep[i]: continue
        src = os.path.realpath(f"{D(sid)}/pool/images/{r['im']}")
        if not os.path.islink(f"{d}/images/{r['im']}"): os.symlink(src, f"{d}/images/{r['im']}")
        W, H = map(float, Image.open(src).size)
        open(f"{d}/labels/{os.path.splitext(r['im'])[0]}.txt", "w").write("".join(
            f"0 {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n" for b in boxes[i]))
    open(f"{D(sid)}_{var}.yaml", "w").write(f"path: {d}\ntrain: images\nval: {D(sid)}/val/images\nnames:\n  0: person\n")

DEV = os.environ.get("DEVICE", "cuda:0")
def gen_sahi(weights, img_dir):
    """The paper's teacher (cr_eval.gen_sahi: 320 px tiles, overlap 0.2, full-frame pass, NMS IoU 0.5, conf >= 0.001) on DEV."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=weights, confidence_threshold=0.001, device=DEV)
    rec = []
    for p in sorted(glob.glob(f"{img_dir}/*.jpg")):
        W, H = Image.open(p).size; g = []; lf = p.replace("/images/", "/labels/")[:-4] + ".txt"
        if os.path.exists(lf):
            for line in open(lf):
                q = line.split()
                if len(q) == 5:
                    cx, cy, w, h = map(float, q[1:]); g.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        r = get_sliced_prediction(p, dm, slice_height=320, slice_width=320, overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                  postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5, verbose=0)
        rec.append({"im": os.path.basename(p), "gt": np.array(g).reshape(-1, 4), "det": np.array([[o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy]
                    for o in r.object_prediction_list]).reshape(-1, 4), "conf": np.array([o.score.value for o in r.object_prediction_list]), "gain": 1.0})
    return rec

def prep():
    from ultralytics import YOLO
    m0 = YOLO(BASELINE); q = {}
    for sid in SIDS:
        cf = f"{C}/teacher_s{sid}_pool.pkl"
        if not os.path.exists(cf): pickle.dump(gen_sahi(BASELINE, f"{D(sid)}/pool/images"), open(cf, "wb"))
        rec = sorted(pickle.load(open(cf, "rb")), key=lambda r: r["im"])
        o = [np.argsort(-r["conf"], kind="stable") for r in rec]; D55 = [r["det"][k][r["conf"][k] >= 0.55] for r, k in zip(rec, o)]
        write_set(sid, "t055", rec, D55); write_set(sid, "t055ne", rec, D55, [len(d) > 0 for d in D55])
        # supervised reference: the labelled pool frames with their real labels
        d = f"{D(sid)}/pool_lab"; os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
        for p in frames(sid, "pool"):
            lf = p.replace("/images/", "/labels/")[:-4] + ".txt"
            if os.path.exists(lf):
                b = os.path.basename(p)
                if not os.path.islink(f"{d}/images/{b}"): os.symlink(os.path.realpath(p), f"{d}/images/{b}")
                if not os.path.exists(f"{d}/labels/{b[:-4]}.txt"): os.symlink(lf, f"{d}/labels/{b[:-4]}.txt")
        open(f"{D(sid)}_sup.yaml", "w").write(f"path: {d}\ntrain: images\nval: {D(sid)}/val/images\nnames:\n  0: person\n")
        cf = f"{C}/base_s{sid}.pkl"
        if not os.path.exists(cf):
            pickle.dump({p: pred_dev(m0, arrays(frames(sid, p))) for p in ("val", "test")}, open(cf, "wb"))
        lab = {os.path.basename(f)[:-4] for f in glob.glob(f"{D(sid)}/pool/labels/*.txt")}
        tp = nb = ng = 0
        for r, d55 in zip(rec, D55):
            if os.path.splitext(r["im"])[0] not in lab: continue
            G = r["gt"]; nb += len(d55); ng += len(G)
            if len(d55) and len(G):
                M = E.iou_mat(d55, G); used = np.zeros(len(G), bool)
                for k in range(len(d55)):
                    row = np.where(used, -1, M[k]); j = int(np.argmax(row))
                    if row[j] >= 0.5: used[j] = True; tp += 1
        q[str(sid)] = {"P": tp / max(nb, 1), "R": tp / max(ng, 1), "frames": len(rec), "frames_with_pl": int(sum(len(d) > 0 for d in D55))}
        print("prep", sid, q[str(sid)], flush=True)
    json.dump(q, open(f"{J}/results/b10_ext_pl.json", "w"), indent=1); print("EXT_PREP_DONE")

def train(npar=4):
    PY = "/opt/conda/bin/python"; running = []
    def launch(name, data, seed, keep):
        if os.path.exists(f"{RUNS}/{name}/DONE"): return
        sp = ", save_period=1" if keep else ""
        code = (f"from ultralytics import YOLO; YOLO('{COCO_N}').train(data='{data}', epochs=50, imgsz=640, batch=16, device='cuda:0', "
                f"workers=4, cache='ram', project='{RUNS}', name='{name}', exist_ok=True, patience=50, seed={seed}, plots=False{sp}); "
                f"open('{RUNS}/{name}/DONE','w').write('ok')")
        p = subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J)
        running.append(p); print(time.strftime("%H:%M:%S"), "launched", name, flush=True)
    def wait(n):
        while True:
            running[:] = [p for p in running if p.poll() is None]
            if len(running) < n: return
            time.sleep(10)
    jobs = [(f"{pre}_s{sid}_seed{k}", f"{D(sid)}_{var}.yaml", k, keep) for pre, var, keep in (("EXm", "t055", True), ("EXne", "t055ne", True), ("EXsup", "sup", False))
            for k in (0, 1, 2) for sid in SIDS]
    for j in jobs: wait(npar); launch(*j)
    wait(1); open(f"{RUNS}/EXT_TRAIN_DONE", "w").write("ok"); print("EXT_TRAIN_DONE", flush=True)

def epochs(wk):
    from ultralytics import YOLO
    os.makedirs(f"{C}/locks", exist_ok=True); A = {}
    jobs = [(pre, sid, k) for pre in ("EXm", "EXne") for k in (0, 1, 2) for sid in SIDS]
    while True:
        done = os.path.exists(f"{RUNS}/EXT_TRAIN_DONE"); pending = 0
        for pre, sid, k in jobs:
            run = f"{RUNS}/{pre}_s{sid}_seed{k}"; cf = f"{C}/{pre}_s{sid}_seed{k}.pkl"
            if os.path.exists(cf): continue
            if not os.path.exists(f"{run}/DONE"): pending += not done; continue
            try: fd = os.open(f"{C}/locks/{pre}_s{sid}_seed{k}", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
            except FileExistsError: continue
            for p in ("val", "test"):
                if (sid, p) not in A: A[(sid, p)] = arrays(frames(sid, p))
            out = {p: {} for p in ("val", "test")}
            for ep in range(50):
                ck = f"{run}/weights/epoch{ep}.pt"
                if not os.path.exists(ck): continue
                m = YOLO(ck)
                for p in out: out[p][ep + 1] = predict(m, A[(sid, p)])
            pickle.dump(out, open(cf + f".tmp{wk}", "wb")); os.replace(cf + f".tmp{wk}", cf)
            for ck in glob.glob(f"{run}/weights/epoch*.pt"): os.remove(ck)
            print(time.strftime("%H:%M:%S"), "cached", pre, sid, k, flush=True)
        if done and pending == 0:
            left = [1 for pre, sid, k in jobs if os.path.exists(f"{RUNS}/{pre}_s{sid}_seed{k}/DONE") and not os.path.exists(f"{C}/{pre}_s{sid}_seed{k}.pkl")]
            if not left: break
        time.sleep(30)
    open(f"{J}/logs/EXT_EPOCHS_DONE_{wk}", "w").write("ok"); print("EXT_EPOCHS_DONE", wk)

def policy():
    from b9_policy import fmatch, ap_idx, guard, draws
    from ultralytics import YOLO
    res = {}
    for sid in SIDS:
        gv, gt = load_gt(frames(sid, "val")), load_gt(frames(sid, "test")); nv = len(gv)
        b = pickle.load(open(f"{C}/base_s{sid}.pkl", "rb")); fm0v, ngtv = fmatch(gv, b["val"]); fm0t, ngtt = fmatch(gt, b["test"])
        base = ap_idx(fm0t, ngtt, np.arange(len(gt))); r = {"base": base, "nv": nv, "nt": len(gt), "people": int(ngtt.sum())}
        for pre in ("EXm", "EXne"):
            V = {"final": [], "pick10": [], "pick10+base": [], "opt": []}
            for k in (0, 1, 2):
                cf = f"{C}/{pre}_s{sid}_seed{k}.pkl"
                if not os.path.exists(cf): continue
                c = pickle.load(open(cf, "rb")); eps = sorted(c["test"])
                ta = {e: ap_idx(*fmatch(gt, c["test"][e]), np.arange(len(gt))) for e in eps}; ta[0] = base
                fmv = {e: fmatch(gv, c["val"][e])[0] for e in eps}; fmv[0] = fm0v
                V["final"].append(ta[eps[-1]]); V["opt"].append(max(ta.values()))
                for wb in (False, True):
                    V["pick10+base" if wb else "pick10"].append(float(np.mean([ta[guard(fmv, ngtv, eps, i, wb)[0]] for i in draws(sid, k, 10, "random", nv)])))
            r[pre] = V
        sup = []
        for k in (0, 1, 2):
            w = f"{RUNS}/EXsup_s{sid}_seed{k}/weights/last.pt"
            if os.path.exists(w): sup.append(ap_idx(*fmatch(gt, predict(YOLO(w), arrays(frames(sid, "test")))), np.arange(len(gt))))
        r["sup"] = sup; res[str(sid)] = r; print("policy", sid, {k: (np.round(np.mean(v["final"]), 1) if isinstance(v, dict) else v) for k, v in r.items() if k in ("base", "EXm", "EXne")}, flush=True)
    json.dump(res, open(f"{J}/results/b10_ext.json", "w"), indent=1, default=float); print("EXT_POLICY_DONE")

if __name__ == "__main__":
    step = sys.argv[1]
    if step == "prep": prep()
    elif step == "train": train(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif step == "epochs": epochs(sys.argv[2])
    elif step == "policy": policy()
