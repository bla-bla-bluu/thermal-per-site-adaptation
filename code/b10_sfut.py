"""Batch 10e: SF-UT, the source-free unbiased (mean) teacher of Hao et al. (ECCV 2024), with the
official code (github.com/EPFL-IMOS/simple-SFOD, detectron2, Faster R-CNN with a VGG16-BN backbone) on the 23 recordings.
Steps (argument), all with the sfut environment's python except policy:
  prep    COCO json files: LLVIP infrared training images (95/5 split as for SF-YOLO, person = category 1); per recording the
          pool frames WITHOUT annotations (target training set) and the validation window with its labels (only for the
          library's end-of-training evaluation log); ImageNet VGG16-BN weights converted with the repository's script
  source  Faster R-CNN VGG16-BN source model on LLVIP (repository's source configuration; 4 images per batch, learning rate
          0.01, 30k iterations, about 10 epochs, instead of 100k single-image iterations, to fit our GPU budget)
  adapt   SF-UT per recording, seed 0 (repository's configuration faster_rcnn_VGG_cityscapes_foggy_adaptive_teacher_
          source_free_single_wq: weak-strong augmentation, pseudo-label threshold 0.8, EMA 0.9996, 4 images per batch,
          learning rate 0.0025) for 2,000 iterations instead of 100k (our pools hold 75 to 1,217 frames; 100k iterations would take
          about three GPU-days per recording); a checkpoint every 100 iterations
  cache   teacher predictions of every checkpoint and of the source model on the validation and test windows
          (score >= 0.001, NMS 0.5, at most 300 boxes)
  policy  final checkpoint, ten-frame guard (with and without the source model) and oracle with the paper's matcher;
          writes results/b10_sfut.json"""
import os, sys, glob, json, pickle, time, subprocess, random
import numpy as np
J = os.environ.get("JROOT", "/path/to/domain/journal"); SR = f"{J}/baselines/simple-SFOD"; PYU = os.environ.get("PYU", "/path/to/envs/sfut/bin/python")
W = f"{J}/sfut"; os.makedirs(W, exist_ok=True); ITERS = int(os.environ.get("SFUT_ITERS", "2000")); CKP = 100
# SFUT_SCHED=epoch: the official number of passes over the target data instead of 2,000 iterations. The repository runs
# 100,000 iterations of 4 target images on the 2,975 Foggy Cityscapes training images, i.e. 134.45 passes; here each
# recording gets round(134.45 * pool / 4) iterations, with 20 checkpoints. Separate folders (adapt_epoch, cache/sfut_epoch).
SCHED = os.environ.get("SFUT_SCHED", "fixed")
# SFUT_LRSTEPS=official: the repository's learning-rate steps (gamma 0.1) at 60, 80 and 90% of the run instead of a constant rate; separate folders (adapt_epoch_lr, cache/sfut_epoch_lr)
LRS = os.environ.get("SFUT_LRSTEPS", ""); VAR = ("" if SCHED == "fixed" else "_epoch") + ("_lr" if LRS else "")
def steps_for(it): return f"({round(.6 * it)}, {round(.8 * it)}, {round(.9 * it)}, {round(3.6 * it)})" if LRS else f"({it * 10},)"
def iters_for(sid):
    if SCHED == "fixed": return ITERS, CKP
    n = len(glob.glob(f"{sdir(sid, 'pool')}/images/*.jpg")); it = int(round(100000 * 4 / 2975 * n / 4)); return it, max(100, it // 20)
sys.path.insert(0, f"{J}/code"); os.environ.setdefault("JDOMAIN", "all")
from jcommon import scenarios, sdir
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
SIDS = [s for s, *_ in scenarios()]

def coco(img_paths, with_ann):
    from PIL import Image
    ims, anns = [], []
    for i, p in enumerate(img_paths):
        w, h = Image.open(p).size; ims.append({"id": i + 1, "file_name": os.path.realpath(p), "width": w, "height": h})
        lf = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if with_ann and os.path.exists(lf):
            for l in open(lf):
                q = l.split()
                if len(q) == 5 and q[0] == "0":
                    cx, cy, bw, bh = map(float, q[1:])
                    anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": 1, "iscrowd": 0, "area": bw * w * bh * h,
                                 "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]})
    return {"images": ims, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}

def prep():
    ims = sorted(glob.glob("/path/to/domain/yolo_cr_llvip/train/images/*.jpg")); rnd = random.Random(0); rnd.shuffle(ims); nv = len(ims) // 20
    json.dump(coco(ims[nv:], True), open(f"{W}/llvip_train.json", "w")); json.dump(coco(ims[:nv], True), open(f"{W}/llvip_val.json", "w"))
    for sid in SIDS:
        t = tag(sid)
        json.dump(coco(sorted(glob.glob(f"{sdir(sid, 'pool')}/images/*.jpg")), False), open(f"{W}/{t}_pool.json", "w"))
        json.dump(coco(sorted(glob.glob(f"{sdir(sid, 'val')}/images/*.jpg")), True), open(f"{W}/{t}_val.json", "w"))
    if not os.path.exists(f"{W}/vgg16_bn.pkl"):
        os.system(f"curl -s -L -o {W}/vgg16_bn.pth https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")
        src = open(f"{SR}/convert_pretrained_model/convert_vgg_bn.py").read()
        src = src.replace('"/cluster/scratch/username/simple-SFOD/vgg16_bn.pth"', f'"{W}/vgg16_bn.pth"').replace(
            '"/cluster/scratch/username/simple-SFOD/vgg16_bn.pkl"', f'"{W}/vgg16_bn.pkl"')
        open(f"{W}/convert_vgg_bn.py", "w").write(src); subprocess.call([PYU, f"{W}/convert_vgg_bn.py"], stdout=subprocess.DEVNULL)
    print("SFUT_PREP_DONE", os.path.exists(f"{W}/vgg16_bn.pkl"), flush=True)

RUNNER = '''import os, sys
sys.path.insert(0, "{SR}"); os.chdir("{SR}")
from detectron2.data.datasets import register_coco_instances
for item in os.environ["SFUT_DATASETS"].split(","):
    n, js = item.split("|"); register_coco_instances(n, {{}}, js, "/")
from detectron2.engine import default_argument_parser, launch
mod = __import__(os.environ["SFUT_MAIN"])
args = default_argument_parser().parse_args()
launch(mod.main, 1, args=(args,))
'''
def runner():
    open(f"{W}/run_sfut.py", "w").write(RUNNER.format(SR=SR))

def source():
    runner(); out = f"{W}/source"
    if os.path.exists(f"{out}/model_final.pth"): print("source exists"); return
    env = dict(os.environ, SFUT_DATASETS=f"llvip_train|{W}/llvip_train.json,llvip_val|{W}/llvip_val.json", SFUT_MAIN="train_net")
    cmd = [PYU, f"{W}/run_sfut.py", "--config-file", f"{SR}/configs/faster_rcnn_VGG_cityscapes_source_new.yaml",
           "MODEL.WEIGHTS", f"{W}/vgg16_bn.pkl", "MODEL.ROI_HEADS.NUM_CLASSES", "1", "DATASETS.TRAIN", "('llvip_train',)",
           "DATASETS.TRAIN_TARGET", "('llvip_train',)", "DATASETS.TEST", "('llvip_val',)", "SOLVER.IMS_PER_BATCH", "4",
           "SOLVER.BASE_LR", "0.01", "SOLVER.MAX_ITER", "30000", "SOLVER.STEPS", "(20000, 26000)", "SOLVER.CHECKPOINT_PERIOD", "30000",
           "TEST.EVAL_PERIOD", "0", "VIS_PERIOD", "0", "OUTPUT_DIR", out, "SEED", "0", "DATALOADER.NUM_WORKERS", "6"]
    rc = subprocess.call(cmd, env=env, stdout=open(f"{J}/logs/sfut_source.log", "w"), stderr=subprocess.STDOUT)
    print("SFUT_SOURCE_DONE", rc, flush=True)

def adapt(npar=2, seed=0):
    runner(); running = []
    for sid in SIDS:
        t = tag(sid); out = f"{W}/adapt{VAR}/{t}_seed{seed}"; ITERS, CKP = iters_for(sid)
        if os.path.exists(f"{out}/DONE"): continue
        while len([p for p in running if p.poll() is None]) >= npar: time.sleep(15)
        env = dict(os.environ, SFUT_DATASETS=f"tgt_{t}|{W}/{t}_pool.json,val_{t}|{W}/{t}_val.json", SFUT_MAIN="train_net_mt")
        cmd = (f"{PYU} {W}/run_sfut.py --config-file {SR}/configs/faster_rcnn_VGG_cityscapes_foggy_adaptive_teacher_source_free_single_wq.yaml "
               f"MODEL.WEIGHTS {W}/source/model_final.pth MODEL.ROI_HEADS.NUM_CLASSES 1 \"DATASETS.TRAIN_TARGET\" \"('tgt_{t}',)\" "
               f"\"DATASETS.TEST\" \"('val_{t}',)\" SOLVER.MAX_ITER {ITERS} \"SOLVER.STEPS\" \"{steps_for(ITERS)}\" SOLVER.WARMUP_ITERS 200 "
               f"SOLVER.CHECKPOINT_PERIOD {CKP} TEST.EVAL_PERIOD 0 VIS_PERIOD 0 OUTPUT_DIR {out} SEED {seed} DATALOADER.NUM_WORKERS 4 "
               f"&& echo ok > {out}/DONE")
        os.makedirs(out, exist_ok=True)
        running.append(subprocess.Popen(cmd, shell=True, env=env, stdout=open(f"{J}/logs/sfut_adapt_{t}.log", "w"), stderr=subprocess.STDOUT))
        print(time.strftime("%H:%M:%S"), "adapt", t, flush=True)
    for p in running: p.wait()
    print("SFUT_ADAPT_DONE", flush=True)

def cache(seed=0):
    sys.path.insert(0, SR); os.chdir(SR)
    import torch
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    import detectron2.data.transforms as T
    from daod.config import add_config
    import importlib
    for mod in ("daod.modeling.meta_arch", "daod.modeling.proposal_generator", "daod.modeling.roi_heads"): importlib.import_module(mod)
    from PIL import Image
    def model_for(cfg_file, weights, teacher):
        cfg = get_cfg(); add_config(cfg); cfg.merge_from_file(cfg_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1; cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001; cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.TEST.DETECTIONS_PER_IMAGE = 300; m = build_model(cfg); m.eval()
        sd = torch.load(weights, map_location="cpu")["model"]
        if teacher: sd = {k[len("modelTeacher."):]: v for k, v in sd.items() if k.startswith("modelTeacher.")}
        m.load_state_dict(sd, strict=False); return m, cfg
    def infer(m, cfg, arrs):
        aug = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST); out = []
        with torch.no_grad():
            for a in arrs:                      # a: BGR uint8
                h, w = a.shape[:2]; im = aug.get_transform(a).apply_image(a)
                r = m([{"image": torch.as_tensor(im.astype("float32").transpose(2, 0, 1)), "height": h, "width": w}])[0]["instances"]
                out.append((r.pred_boxes.tensor.cpu().numpy().reshape(-1, 4), r.scores.cpu().numpy().reshape(-1)))
        return out
    os.makedirs(f"{J}/cache/sfut{VAR}", exist_ok=True)
    cfg_src = f"{SR}/configs/faster_rcnn_VGG_cityscapes_source_new.yaml"
    cfg_ad = f"{SR}/configs/faster_rcnn_VGG_cityscapes_foggy_adaptive_teacher_source_free_single_wq.yaml"
    for sid in SIDS:
        t = tag(sid); arr = {p: [np.asarray(Image.open(q).convert("RGB"))[:, :, ::-1].copy() for q in sorted(glob.glob(f"{sdir(sid, p)}/images/*.jpg"))]
                             for p in ("val", "test")}
        bf = f"{J}/cache/sfut/base_{t}.pkl"
        if not os.path.exists(bf):
            m, cfg = model_for(cfg_src, f"{W}/source/model_final.pth", False); pickle.dump({p: infer(m, cfg, arr[p]) for p in arr}, open(bf, "wb"))
        cf = f"{J}/cache/sfut{VAR}/{t}_seed{seed}.pkl"; out_d = f"{W}/adapt{VAR}/{t}_seed{seed}"; ITERS, _ = iters_for(sid)
        if os.path.exists(cf) or not os.path.exists(f"{out_d}/DONE"): continue
        cks = [c for c in sorted(glob.glob(f"{out_d}/model_0*.pth")) if not c.endswith(f"model_{ITERS - 1:07d}.pth")] + [f"{out_d}/model_final.pth"]
        out = {"val": {}, "test": {}}
        for e, ck in enumerate(cks, 1):
            m, cfg = model_for(cfg_ad, ck, True)
            for p in out: out[p][e] = infer(m, cfg, arr[p])
        pickle.dump(out, open(cf, "wb"))
        for ck in cks[:-1]: os.remove(ck)
        print(time.strftime("%H:%M:%S"), "cached", t, len(cks), flush=True)
    print("SFUT_CACHE_DONE", flush=True)

def policy():
    """Seeds 0-2 where cached; every key is a list over the seeds present (base: the source model)."""
    from b9_policy import fmatch, ap_idx, guard, draws
    res = {}
    for sid in SIDS:
        t = tag(sid); gv = pickle.load(open(f"{J}/cache/b9/gt_{t}_val.pkl", "rb")); gt = pickle.load(open(f"{J}/cache/b9/gt_{t}_test.pkl", "rb")); nv = len(gv)
        b = pickle.load(open(f"{J}/cache/sfut/base_{t}.pkl", "rb")); fm0v, ngtv = fmatch(gv, b["val"])
        base = ap_idx(*fmatch(gt, b["test"]), np.arange(len(gt))); V = {"base": base, "final": [], "opt": [], "pick10": [], "pick10+base": [], "seeds": []}
        for seed in (0, 1, 2):
            cf = f"{J}/cache/sfut{VAR}/{t}_seed{seed}.pkl"
            if not os.path.exists(cf): continue
            c = pickle.load(open(cf, "rb")); eps = sorted(c["test"])
            ta = {e: ap_idx(*fmatch(gt, c["test"][e]), np.arange(len(gt))) for e in eps}; ta[0] = base
            fmv = {e: fmatch(gv, c["val"][e])[0] for e in eps}; fmv[0] = fm0v
            V["final"].append(ta[eps[-1]]); V["opt"].append(max(ta.values())); V["seeds"].append(seed)
            for wb in (False, True):
                V["pick10+base" if wb else "pick10"].append(float(np.mean([ta[guard(fmv, ngtv, eps, i, wb)[0]] for i in draws(sid, seed, 10, "random", nv)])))
        res[t] = V; print("policy", t, round(base, 1), {k: [round(x, 1) for x in v] for k, v in V.items() if k != "base"}, flush=True)
    json.dump(res, open(f"{J}/results/b10_sfut{VAR}.json", "w"), indent=1, default=float); print("SFUT_POLICY_DONE")

if __name__ == "__main__":
    s = sys.argv[1]
    if s == "prep": prep()
    elif s == "source": source()
    elif s == "adapt": adapt(int(sys.argv[2]) if len(sys.argv) > 2 else 2, int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0)
    elif s == "cache": cache(int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0)
    elif s == "policy": policy()
