"""Batch 16: Cityscapes -> Foggy Cityscapes (fog density beta =
0.02, the setting of the source-free detection literature), person class (Cityscapes label id 24; riders, label 25, are
background), in the paper's protocol:
  source   YOLOv8-Nano from COCO, trained on the 2,975 clear Cityscapes training images (person boxes from gtFine instance
           ids), 50 epochs, 1280 px (Cityscapes frames are 2048 x 1024; the literature resizes them to a 600-px short side)
  target   pool = the 2,975 foggy training images (unlabelled); test = the 500 foggy validation images (the literature's
           test set); guard frames = 100 foggy training images drawn once (seed 0), whose labels only the guard uses
  teacher  source + SAHI: 640-px tiles (twofold enlargement, as in the paper), overlap 0.2, full frame, NMS 0.5; tau 0.55
  students YOLOv8-Nano from COCO on the pseudo-labels, 50 epochs, 1280 px, batch 8, seeds 0-2, every epoch kept; variants
           t055 (empty frames kept) and t055ne (left out)
Steps: unzip | build | source | prep | train <npar> | epochs <worker> | policy. Writes results/b16_cityscapes.json."""
import os, sys, glob, json, pickle, time, subprocess, zipfile
import numpy as np
from PIL import Image
J = "/path/to/domain/journal"; RAW = f"{J}/external/cityscapes"; X = f"{RAW}/x"; D = f"{J}/data/cityscapes"
C = f"{J}/cache/cityscapes"; RUNS = f"{J}/runs_cs"; PY = "/opt/conda/bin/python"; IMGSZ = 1280; TAU = 0.55
for d in (D, C, RUNS): os.makedirs(d, exist_ok=True)
def unzip():
    os.makedirs(X, exist_ok=True)
    with zipfile.ZipFile(f"{RAW}/gtFine_trainvaltest.zip") as z: z.extractall(X, [n for n in z.namelist() if n.endswith("instanceIds.png") and "/test/" not in n])
    with zipfile.ZipFile(f"{RAW}/leftImg8bit_trainvaltest.zip") as z: z.extractall(X, [n for n in z.namelist() if "/train/" in n and n.endswith(".png")])
    with zipfile.ZipFile(f"{RAW}/leftImg8bit_trainvaltest_foggy.zip") as z: z.extractall(X, [n for n in z.namelist() if n.endswith("beta_0.02.png") and "/test/" not in n])
    print("CS_UNZIP_DONE", flush=True)
def boxes_from_ids(p):
    a = np.asarray(Image.open(p)); out = []
    for i in np.unique(a):
        if i >= 1000 and i // 1000 == 24:
            ys, xs = np.nonzero(a == i); out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return out
def write(dst, img, bxs, W=2048, H=1024):
    os.makedirs(f"{dst}/images", exist_ok=True); os.makedirs(f"{dst}/labels", exist_ok=True); b = os.path.basename(img).replace(".png", ".png")
    if not os.path.islink(f"{dst}/images/{b}"): os.symlink(img, f"{dst}/images/{b}")
    if bxs is not None:
        open(f"{dst}/labels/{b[:-4]}.txt", "w").write("".join(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n" for x1, y1, x2, y2 in bxs if x2 - x1 >= 2 and y2 - y1 >= 2))
def build():
    gt = {os.path.basename(p).replace("_gtFine_instanceIds.png", ""): p for p in glob.glob(f"{X}/gtFine/*/*/*_instanceIds.png")}
    for p in sorted(glob.glob(f"{X}/leftImg8bit/train/*/*.png")):                     # clear source training images
        write(f"{D}/src_train", p, boxes_from_ids(gt[os.path.basename(p).replace("_leftImg8bit.png", "")]))
    fog = {sp: sorted(glob.glob(f"{X}/leftImg8bit_foggy/{sp}/*/*beta_0.02.png")) for sp in ("train", "val")}
    key = lambda p: os.path.basename(p).replace("_leftImg8bit_foggy_beta_0.02.png", "")
    guard = set(np.random.default_rng(0).choice(len(fog["train"]), 100, replace=False).tolist())
    for i, p in enumerate(fog["train"]):
        write(f"{D}/pool", p, boxes_from_ids(gt[key(p)]))                              # labels kept only for diagnostics
        if i in guard: write(f"{D}/guard", p, boxes_from_ids(gt[key(p)]))
    for p in fog["val"]: write(f"{D}/test", p, boxes_from_ids(gt[key(p)]))
    for p in sorted(glob.glob(f"{X}/leftImg8bit/train/*/*.png"))[:0]: pass
    open(f"{D}/src.yaml", "w").write(f"path: {D}\ntrain: src_train/images\nval: test/images\nnames:\n  0: person\n")   # val only logged
    print("CS_BUILD_DONE", {k: len(glob.glob(f"{D}/{k}/images/*.png")) for k in ("src_train", "pool", "guard", "test")}, flush=True)
def source():
    if os.path.exists(f"{RUNS}/source/DONE"): return
    code = (f"from ultralytics import YOLO; YOLO('{J}/yolov8n.pt').train(data='{D}/src.yaml', epochs=50, imgsz={IMGSZ}, batch=8, device='cuda:0', workers=6, "
            f"project='{RUNS}', name='source', exist_ok=True, seed=0, plots=False, val=False); open('{RUNS}/source/DONE','w').write('ok')")
    subprocess.call([PY, "-c", code], stdout=open(f"{J}/logs/cs_source.log", "w"), stderr=subprocess.STDOUT, cwd=J); print("CS_SOURCE_DONE", flush=True)
SRC = f"{RUNS}/source/weights/last.pt"
def predict(m, paths):
    out = []
    for i in range(0, len(paths), 16):
        for r in m.predict(paths[i:i + 16], imgsz=IMGSZ, conf=0.001, iou=0.5, max_det=300, verbose=False, device="cuda:0"):
            out.append((r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy().reshape(-1)))
    return out
def prep():
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO
    tf = f"{C}/teacher_pool.pkl"
    if not os.path.exists(tf):
        dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=SRC, confidence_threshold=0.001, device="cuda:0", image_size=IMGSZ)
        rec = []
        for p in sorted(glob.glob(f"{D}/pool/images/*.png")):
            r = get_sliced_prediction(p, dm, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                      postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5, verbose=0)
            ob = r.object_prediction_list
            rec.append({"im": os.path.basename(p), "det": np.array([[o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy] for o in ob]).reshape(-1, 4), "conf": np.array([o.score.value for o in ob])})
        pickle.dump(rec, open(tf, "wb"))
    rec = pickle.load(open(tf, "rb"))
    for var in ("t055", "t055ne"):
        d = f"{D}/pl_{var}"; os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
        for r in rec:
            k = r["conf"] >= TAU; B = r["det"][k]
            if var == "t055ne" and len(B) == 0: continue
            src = os.path.realpath(f"{D}/pool/images/{r['im']}")
            if not os.path.islink(f"{d}/images/{r['im']}"): os.symlink(src, f"{d}/images/{r['im']}")
            open(f"{d}/labels/{r['im'][:-4]}.txt", "w").write("".join(f"0 {(b[0]+b[2])/2/2048:.6f} {(b[1]+b[3])/2/1024:.6f} {(b[2]-b[0])/2048:.6f} {(b[3]-b[1])/1024:.6f}\n" for b in B))
        open(f"{D}/{var}.yaml", "w").write(f"path: {d}\ntrain: images\nval: {D}/test/images\nnames:\n  0: person\n")
    bf = f"{C}/base.pkl"
    if not os.path.exists(bf):
        m = YOLO(SRC); pickle.dump({p: predict(m, sorted(glob.glob(f"{D}/{p}/images/*.png"))) for p in ("guard", "test")}, open(bf, "wb"))
    print("CS_PREP_DONE", "boxes/frame", round(np.mean([(r["conf"] >= TAU).sum() for r in rec]), 2), "empty frames", sum((r["conf"] >= TAU).sum() == 0 for r in rec), flush=True)
JOBS = [(f"CS{var}_seed{k}", var, k) for var in ("t055", "t055ne") for k in (0, 1, 2)]
def train(npar):
    running = []
    for name, var, k in JOBS:
        if os.path.exists(f"{RUNS}/{name}/DONE"): continue
        while len([p for p in running if p.poll() is None]) >= npar: time.sleep(20)
        code = (f"from ultralytics import YOLO; YOLO('{J}/yolov8n.pt').train(data='{D}/{var}.yaml', epochs=50, imgsz={IMGSZ}, batch=8, device='cuda:0', workers=6, "
                f"project='{RUNS}', name='{name}', exist_ok=True, seed={k}, plots=False, val=False, save_period=1); open('{RUNS}/{name}/DONE','w').write('ok')")
        running.append(subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J))
        print(time.strftime("%H:%M:%S"), "launched", name, flush=True)
    for p in running: p.wait()
    open(f"{RUNS}/CS_TRAIN_DONE", "w").write("ok"); print("CS_TRAIN_DONE", flush=True)
def epochs(wk):
    from ultralytics import YOLO
    os.makedirs(f"{C}/locks", exist_ok=True); P = {p: sorted(glob.glob(f"{D}/{p}/images/*.png")) for p in ("guard", "test")}
    while True:
        done = os.path.exists(f"{RUNS}/CS_TRAIN_DONE"); pending = 0
        for name, var, k in JOBS:
            cf = f"{C}/{name}.pkl"
            if os.path.exists(cf): continue
            if not os.path.exists(f"{RUNS}/{name}/DONE"): pending += 1; continue
            try: os.mkdir(f"{C}/locks/{name}")
            except FileExistsError: continue
            out = {p: {} for p in P}
            for ep in range(50):
                ck = f"{RUNS}/{name}/weights/epoch{ep}.pt"
                if not os.path.exists(ck): continue
                m = YOLO(ck)
                for p in out: out[p][ep + 1] = predict(m, P[p])
            pickle.dump(out, open(cf + ".tmp", "wb")); os.replace(cf + ".tmp", cf)
            for ck in glob.glob(f"{RUNS}/{name}/weights/epoch*.pt"): os.remove(ck)
            print(time.strftime("%H:%M:%S"), "cached", name, flush=True)
        if done and pending == 0: break
        time.sleep(60)
    print("CS_EPOCHS_DONE", wk, flush=True)
def policy():
    sys.path.insert(0, f"{J}/code"); os.environ.setdefault("JDOMAIN", "all")
    from b9_policy import fmatch, ap_idx, guard
    from selection_study_helpers import load_gt
    gv = load_gt(sorted(glob.glob(f"{D}/guard/images/*.png"))); gt = load_gt(sorted(glob.glob(f"{D}/test/images/*.png"))); nv, nt = len(gv), len(gt)
    b = pickle.load(open(f"{C}/base.pkl", "rb")); fm0v, ngtv = fmatch(gv, b["guard"]); base = ap_idx(*fmatch(gt, b["test"]), np.arange(nt))
    res = {"base": base, "test_people": int(sum(len(g) for g in gt))}
    for var in ("t055", "t055ne"):
        V = {"final": [], "pick10": [], "pick10+base": [], "opt": [], "curve": []}
        for k in (0, 1, 2):
            cf = f"{C}/CS{var}_seed{k}.pkl"
            if not os.path.exists(cf): continue
            c = pickle.load(open(cf, "rb")); eps = sorted(c["test"]); ta = {e: ap_idx(*fmatch(gt, c["test"][e]), np.arange(nt)) for e in eps}; ta[0] = base
            fmv = {e: fmatch(gv, c["guard"][e])[0] for e in eps}; fmv[0] = fm0v
            V["final"].append(ta[eps[-1]]); V["opt"].append(max(ta.values())); V["curve"].append([ta[e] for e in eps])
            rng = np.random.default_rng([16, k]); D10 = [rng.choice(nv, 10, replace=False) for _ in range(50)]
            for wb in (False, True):
                V["pick10+base" if wb else "pick10"].append(float(np.mean([ta[guard(fmv, ngtv, eps, i, wb)[0]] for i in D10])))
        res[var] = V; print(var, {kk: (round(float(np.mean(v)), 1) if v and not isinstance(v[0], list) else None) for kk, v in V.items()}, flush=True)
    json.dump(res, open(f"{J}/results/b16_cityscapes.json", "w"), indent=1, default=float); print("CS_POLICY_DONE base", round(base, 1))
if __name__ == "__main__":
    s = sys.argv[1]
    {"unzip": unzip, "build": build, "source": source, "prep": prep, "policy": policy}.get(s, lambda: None)()
    if s == "train": train(int(sys.argv[2]))
    if s == "epochs": epochs(sys.argv[2])
