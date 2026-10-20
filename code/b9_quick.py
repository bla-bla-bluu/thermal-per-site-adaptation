"""Batch 9b: analyses that need no training.
  adabn  AdaBN baseline: the source detector with its batch-normalisation statistics re-estimated on the pool frames of
         the recording (one pass, cumulative average, the letterbox preprocessing of YOLO.predict); val/test predictions
  tval   the teacher (sliced inference, as for the pseudo-labels) on the validation frames
  sup    supervised reference: test predictions of the three seeds (for AP75 and AP50:95)
Writes cache/b9/{adabn,tval,sup}_*.pkl and results/b9_quick.json."""
import os, glob, json, pickle
import numpy as np
import torch
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from selection_study_helpers import load_gt, predict
os.makedirs(f"{J}/cache/b9", exist_ok=True)
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def frames(sid, part): return sorted(glob.glob(f"{sdir(sid, part)}/images/*.jpg"))
def arrays(ps): return [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1] for p in ps]
def ap50(gt, det):
    d, g = E.match([{"gt": a, "det": b, "conf": c, "gain": 1.0} for a, (b, c) in zip(gt, det)]); return 100 * E.summarize(d, len(g))["AP50"]

def adabn(pool_paths):
    m = YOLO(BASELINE); net = m.model.cuda().float()
    for mod in net.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.reset_running_stats(); mod.momentum = None        # cumulative average over all pool frames
    net.train(); lb = LetterBox(new_shape=(640, 640), auto=True, stride=32)
    with torch.no_grad():
        for i in range(0, len(pool_paths), 16):
            ims = [lb(image=a) for a in arrays(pool_paths[i:i + 16])]
            x = np.stack([im[..., ::-1].transpose(2, 0, 1) for im in ims])          # BGR -> RGB, HWC -> CHW
            net(torch.from_numpy(np.ascontiguousarray(x)).cuda().float() / 255)
    net.eval(); return m

SC = [s for s, *_ in scenarios()]; res = {"adabn": {}, "tval": {}, "sup": {}}
for sid in SC:
    t = tag(sid); vp, tp = frames(sid, "val"), frames(sid, "test"); va, ta = arrays(vp), arrays(tp)
    gv, gt = load_gt(vp), load_gt(tp)
    cf = f"{J}/cache/b9/adabn_{t}.pkl"
    if not os.path.exists(cf):
        m = adabn(frames(sid, "pool")); pickle.dump({"val": predict(m, va), "test": predict(m, ta)}, open(cf, "wb"))
    c = pickle.load(open(cf, "rb")); res["adabn"][t] = {"val": ap50(gv, c["val"]), "test": ap50(gt, c["test"])}
    cf = f"{J}/cache/b9/tval_{t}.pkl"
    if not os.path.exists(cf):
        pickle.dump(E.gen_sahi(BASELINE, 320, img_dir=f"{sdir(sid, 'val')}/images"), open(cf, "wb"))
    rec = pickle.load(open(cf, "rb")); res["tval"][t] = ap50(gv, [(r["det"], r["conf"]) for r in rec])
    cf = f"{J}/cache/b9/sup_{t}.pkl"
    if not os.path.exists(cf):
        pickle.dump({k: predict(YOLO(f"{J}/runs/SUP_{t}_seed{k}/weights/last.pt"), ta) for k in (0, 1, 2)}, open(cf, "wb"))
    c = pickle.load(open(cf, "rb")); res["sup"][t] = {k: ap50(gt, c[k]) for k in c}
    print("done", t, res["adabn"][t], round(res["tval"][t], 1), {k: round(v, 1) for k, v in res["sup"][t].items()}, flush=True)
json.dump(res, open(f"{J}/results/b9_quick.json", "w"), indent=1)
print("B9_QUICK_DONE")
