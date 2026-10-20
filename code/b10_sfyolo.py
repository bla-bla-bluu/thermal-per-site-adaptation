"""Batch 10d: SF-YOLO (Varailhon et al., ECCV-W 2024; official code github.com/vs-cv/sf-yolo, YOLOv5)
on the 23 recordings, in the paper's per-recording protocol.
Steps (argument):
  prep    YOLOv5 data: LLVIP infrared training images split 95/5 (deterministic) for the source model; per recording a data
          yaml (train = pool frames, val = validation window), a content folder (pool frames) and the mean pool image as
          the style image of the Target Augmentation Module (TAM), as in the official recipe (mean target image)
  source  YOLOv5-Nano source model on LLVIP (official train_source.py, COCO-pretrained yolov5n.pt, 50 epochs, 640 px,
          batch 16, default hyperparameters); nano to match the paper's detectors
  tam     TAM per recording (official TargetAugment_train/train.py, default settings except --max_iter, see TAM_ITERS)
  adapt   SF-YOLO per recording and seed 0-2 (official train_sf-yolo.py with the README settings: 60 epochs, batch 16,
          style_add_alpha 0.4, SSM_alpha 0.5, teacher EMA 0.999, conf_thres 0.4; image size 640 to match the paper);
          every epoch's teacher saved (--save-period 1); AutoAnchor and per-epoch validation off, and the
          pool folder has no label files, so no target label reaches the adaptation
  cache   per-epoch teacher predictions on the validation and test windows (YOLOv5 inference: letterbox 640, conf >=
          0.001, NMS IoU 0.5, at most 300 boxes), then the epoch checkpoints are deleted
  policy  final epoch, ten-frame guard (with and without the source model) and oracle with the paper's matcher;
          writes results/b10_sfyolo.json
The steps source/tam/adapt/cache run with the sf-yolo environment's python; prep/policy with /opt/conda."""
import os, sys, glob, json, pickle, time, subprocess, random
import numpy as np
J = os.environ.get("JROOT", "/path/to/domain/journal"); SF = f"{J}/baselines/sf-yolo"; PYS = os.environ.get("PYS", "/path/to/envs/sfyolo/bin/python")
SEEDS = [int(x) for x in os.environ.get("SFY_SEEDS", "0,1,2").split(",")]   # a second GPU box runs some seeds (marker REMOTE here)
W = f"{J}/sfyolo"; os.makedirs(W, exist_ok=True); TAM_ITERS = int(os.environ.get("TAM_ITERS", "40000"))
# the 40k-iteration style modules are the paper's main SF-YOLO runs; other schedules (the default 160k) keep their own folders
VAR = "" if TAM_ITERS == 40000 else f"_tam{TAM_ITERS // 1000}k"
TAMD, ADPD, CACHED = f"{W}/tam{VAR}", f"{W}/adapt{VAR}", f"{J}/cache/sfyolo{VAR}"
sys.path.insert(0, f"{J}/code"); os.environ.setdefault("JDOMAIN", "all")
from jcommon import scenarios, sdir
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
SIDS = [s for s, *_ in scenarios()]

def prep():
    from PIL import Image
    ims = sorted(glob.glob("/path/to/domain/yolo_cr_llvip/train/images/*.jpg")); rnd = random.Random(0); rnd.shuffle(ims)
    nv = len(ims) // 20; open(f"{W}/llvip_val.txt", "w").write("\n".join(ims[:nv]) + "\n"); open(f"{W}/llvip_train.txt", "w").write("\n".join(ims[nv:]) + "\n")
    open(f"{W}/llvip.yaml", "w").write(f"path: {W}\ntrain: {W}/llvip_train.txt\nval: {W}/llvip_val.txt\nnc: 1\nnames: ['person']\n")
    for sid in SIDS:
        t = tag(sid); pool = sorted(glob.glob(f"{sdir(sid, 'pool')}/images/*.jpg")); cd = f"{W}/tam/{t}/content"; sd = f"{W}/tam/{t}/style"
        os.makedirs(cd, exist_ok=True); os.makedirs(sd, exist_ok=True)
        for p in pool:
            q = f"{cd}/{os.path.basename(p)}"
            if not os.path.islink(q): os.symlink(os.path.realpath(p), q)
        acc = None
        for p in pool:
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64); acc = a if acc is None else acc + a
        Image.fromarray((acc / len(pool)).round().astype(np.uint8)).save(f"{sd}/mean.jpg", quality=95)
        nl = f"{W}/pool_nolabel/{t}/images"; os.makedirs(nl, exist_ok=True); os.makedirs(nl.replace("/images", "/labels"), exist_ok=True)
        for p in pool:      # pool frames with EMPTY label files (YOLOv5 needs label files to train): no target label can leak
            q = f"{nl}/{os.path.basename(p)}"
            if not os.path.islink(q): os.symlink(os.path.realpath(p), q)
            open(q.replace("/images/", "/labels/")[:-4] + ".txt", "w").close()
        # YOLOv5's loader refuses a training set whose label files are all empty; one fixed dummy box in the first frame
        # satisfies it. SF-YOLO's loss uses only the teacher's pseudo-labels, so the dummy box never enters training.
        open(f"{nl.replace('/images', '/labels')}/{os.path.basename(pool[0])[:-4]}.txt", "w").write("0 0.5 0.5 0.01 0.01\n")
        for c in glob.glob(f"{W}/pool_nolabel/{t}/*.cache"): os.remove(c)
        open(f"{W}/{t}.yaml", "w").write(f"path: {W}\ntrain: {nl}\nval: {sdir(sid, 'val')}/images\nnc: 1\nnames: ['person']\n")
    print("SFYOLO_PREP_DONE", flush=True)

def run(cmd, log, cwd):
    with open(log, "w") as f: return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)

def source():
    if os.path.exists(f"{W}/source/DONE"): return
    rc = run([PYS, "train_source.py", "--data", f"{W}/llvip.yaml", "--weights", "yolov5n.pt", "--epochs", "50", "--batch-size", "16", "--imgsz", "640",
              "--device", "0", "--workers", "8", "--project", W, "--name", "source", "--exist-ok", "--seed", "0", "--noplots"], f"{J}/logs/sfyolo_source.log", SF)
    if rc == 0: open(f"{W}/source/DONE", "w").write("ok")
    print("SFYOLO_SOURCE_DONE", rc, flush=True)

def tam(npar=3):
    running = []
    for sid in SIDS:
        t = tag(sid); sd = f"{TAMD}/{t}/models"
        if os.path.exists(f"{sd}/decoder_iter_{TAM_ITERS}.pth"): continue
        while len([p for p in running if p.poll() is None]) >= npar: time.sleep(10)
        running.append(subprocess.Popen([PYS, "train.py", "--scenario_name", "city2foggy", "--content_dir", f"{W}/tam/{t}/content",
                        "--style_dir", f"{W}/tam/{t}/style", "--vgg", "pre_trained/vgg16_ori.pth", "--save_dir", sd, "--n_threads", "4",
                        "--device", "0", "--max_iter", str(TAM_ITERS)], cwd=f"{SF}/TargetAugment_train",
                        stdout=open(f"{J}/logs/sfyolo_tam_{t}.log", "w"), stderr=subprocess.STDOUT))
        print(time.strftime("%H:%M:%S"), "tam", t, flush=True)
    for p in running: p.wait()
    print("SFYOLO_TAM_DONE", sum(os.path.exists(f"{TAMD}/{tag(s)}/models/decoder_iter_{TAM_ITERS}.pth") for s in SIDS), flush=True)

def adapt(npar=3):
    src = f"{W}/source/weights/last.pt"; running = []
    def wait(n):
        while True:
            running[:] = [p for p in running if p.poll() is None]
            if len(running) < n: return
            time.sleep(15)
    for k in SEEDS:
        for sid in SIDS:
            t = tag(sid); name = f"{t}_seed{k}"; m = f"{TAMD}/{t}/models"
            if os.path.exists(f"{ADPD}/{name}/DONE"): continue
            wait(npar)
            cmd = (f"{PYS} train_sf-yolo.py --epochs 60 --batch-size 16 --imgsz 640 --data {W}/{t}.yaml --weights {src} "
                   f"--decoder_path {m}/decoder_iter_{TAM_ITERS}.pth --encoder_path TargetAugment_train/pre_trained/vgg16_ori.pth "
                   f"--fc1 {m}/fc1_iter_{TAM_ITERS}.pth --fc2 {m}/fc2_iter_{TAM_ITERS}.pth --style_add_alpha 0.4 --style_path {W}/tam/{t}/style/mean.jpg "
                   f"--SSM_alpha 0.5 --device 0 --workers 4 --seed {k} --save-period 1 --project {ADPD} --name {name} --exist-ok --noplots --noautoanchor --noval "
                   f"&& echo ok > {ADPD}/{name}/DONE")
            running.append(subprocess.Popen(cmd, shell=True, cwd=SF, stdout=open(f"{J}/logs/sfyolo_adapt_{name}.log", "w"), stderr=subprocess.STDOUT))
            print(time.strftime("%H:%M:%S"), "adapt", name, flush=True)
    wait(1); open(f"{ADPD}/ADAPT_DONE", "w").write("ok"); print("SFYOLO_ADAPT_DONE", flush=True)

def cache(wk, nw):
    sys.path.insert(0, SF)
    import torch
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from PIL import Image
    dev = torch.device("cuda:0"); os.makedirs(CACHED, exist_ok=True)
    def infer(w, arrs):
        m = DetectMultiBackend(w, device=dev, fp16=False); m.eval(); out = []
        for i in range(0, len(arrs), 32):
            ims = [letterbox(a, 640, stride=32, auto=True)[0] for a in arrs[i:i + 32]]
            x = torch.from_numpy(np.ascontiguousarray(np.stack([im[..., ::-1].transpose(2, 0, 1) for im in ims]))).to(dev).float() / 255
            with torch.no_grad(): pred = non_max_suppression(m(x), conf_thres=0.001, iou_thres=0.5, max_det=300)
            for a, p in zip(arrs[i:i + 32], pred):
                p = p.clone(); p[:, :4] = scale_boxes(x.shape[2:], p[:, :4], a.shape[:2]).round()
                out.append((p[:, :4].cpu().numpy().reshape(-1, 4), p[:, 4].cpu().numpy().reshape(-1)))
        return out
    jobs = [(sid, k) for k in SEEDS for sid in SIDS]; A = {}
    for n, (sid, k) in enumerate(jobs):
        if n % nw != wk: continue
        t = tag(sid); name = f"{t}_seed{k}"; cf = f"{CACHED}/{name}.pkl"
        if os.path.exists(cf) or not os.path.exists(f"{ADPD}/{name}/DONE") or os.path.exists(f"{ADPD}/{name}/REMOTE"): continue
        for p in ("val", "test"):
            if (sid, p) not in A: A[(sid, p)] = [np.asarray(Image.open(q).convert("RGB"))[:, :, ::-1] for q in sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg"))]
        cks = sorted(glob.glob(f"{ADPD}/{name}/weights/epoch*_teacher.pt"), key=lambda q: int(os.path.basename(q)[5:].split("_")[0]))
        out = {"val": {}, "test": {}}
        for ck in cks:
            e = int(os.path.basename(ck)[5:].split("_")[0]) + 1
            for p in out: out[p][e] = infer(ck, A[(sid, p)])
        pickle.dump(out, open(cf, "wb"))
        for ck in cks: os.remove(ck)
        print(time.strftime("%H:%M:%S"), "cached", name, len(cks), flush=True)
    if not VAR and not os.path.exists(f"{J}/cache/sfyolo/base_s01.pkl") and wk == 0:     # the YOLOv5-Nano source model itself
        src = f"{W}/source/weights/last.pt"
        for sid in SIDS:
            arr = {p: [np.asarray(Image.open(q).convert("RGB"))[:, :, ::-1] for q in sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg"))] for p in ("val", "test")}
            pickle.dump({p: infer(src, arr[p]) for p in arr}, open(f"{J}/cache/sfyolo/base_{tag(sid)}.pkl", "wb"))
    print("SFYOLO_CACHE_DONE", wk, flush=True)

def policy():
    from b9_policy import fmatch, ap_idx, guard, draws
    res = {}
    for sid in SIDS:
        t = tag(sid); gv = pickle.load(open(f"{J}/cache/b9/gt_{t}_val.pkl", "rb")); gt = pickle.load(open(f"{J}/cache/b9/gt_{t}_test.pkl", "rb")); nv = len(gv)
        b = pickle.load(open(f"{J}/cache/sfyolo/base_{t}.pkl", "rb")); fm0v, ngtv = fmatch(gv, b["val"])
        base = ap_idx(*fmatch(gt, b["test"]), np.arange(len(gt))); V = {"base": base, "final": [], "pick10": [], "pick10+base": [], "opt": []}
        for k in (0, 1, 2):
            cf = f"{CACHED}/{t}_seed{k}.pkl"
            if not os.path.exists(cf): continue
            c = pickle.load(open(cf, "rb")); eps = sorted(c["test"])
            ta = {e: ap_idx(*fmatch(gt, c["test"][e]), np.arange(len(gt))) for e in eps}; ta[0] = base
            fmv = {e: fmatch(gv, c["val"][e])[0] for e in eps}; fmv[0] = fm0v
            V["final"].append(ta[eps[-1]]); V["opt"].append(max(ta.values()))
            for wb in (False, True):
                V["pick10+base" if wb else "pick10"].append(float(np.mean([ta[guard(fmv, ngtv, eps, i, wb)[0]] for i in draws(sid, k, 10, "random", nv)])))
        res[t] = V; print("policy", t, round(base, 1), {k: round(float(np.mean(v)), 1) for k, v in V.items() if isinstance(v, list) and v}, flush=True)
    json.dump(res, open(f"{J}/results/b10_sfyolo{VAR}.json", "w"), indent=1, default=float); print("SFYOLO_POLICY_DONE")

if __name__ == "__main__":
    s = sys.argv[1]
    if s == "prep": prep()
    elif s == "source": source()
    elif s == "tam": tam()
    elif s == "adapt": adapt(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif s == "cache": cache(int(sys.argv[2]), int(sys.argv[3]))
    elif s == "policy": policy()
