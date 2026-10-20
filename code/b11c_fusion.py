"""Batch 11c: a label-free safeguard by fusion. Instead of choosing between the source detector theta0 and the
student, deploy both: weighted box fusion (as b11_methods.wbf, IoU 0.55, fused score = mean of the member scores, a box seen by
one member only keeps half its score) of theta0 and the final-epoch student ("fuse1"), or of theta0 and the fusion of the
epochs 10, 15, ..., 50 ("fuse9"). No labels are used. Evaluated on the test windows with the paper's matcher in every setting
that has caches: the 23 recordings (cache/b9, cache/b6), the other source detectors (cache/b12), MOTChallenge (cache/mot),
CAMEL/BIRDSAI (cache/ext) and Cityscapes -> Foggy Cityscapes (cache/cityscapes). Writes results/b11_fusion.json."""
import os, sys, json, glob, pickle
import numpy as np
from multiprocessing import Pool
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
from b9_policy import fmatch, ap_idx, L
from b11_methods import wbf
from selection_study_helpers import load_gt
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
def ap(gt, det): return ap_idx(*fmatch(gt, det), np.arange(len(gt)))
def fuse_frames(models):
    """models: list over models of per-frame (det, conf); returns the fused per-frame list; K = number of models."""
    return [wbf([m[i] for m in models], len(models)) for i in range(len(models[0]))]
def one_setting(gt, base_test, student_tests):
    """student_tests: list over seeds of {epoch: per-frame dets}; returns dict of per-seed APs."""
    out = {"base": ap(gt, base_test), "final": [], "fuse1": [], "fuse9": []}
    for c in student_tests:
        eps = sorted(c); out["final"].append(ap(gt, c[eps[-1]]))
        out["fuse1"].append(ap(gt, fuse_frames([base_test, c[eps[-1]]])))
        sel = [e for e in eps if e >= 10 and e % 5 == 0]; ens = fuse_frames([c[e] for e in sel])
        out["fuse9"].append(ap(gt, fuse_frames([base_test, ens])))
    return out
def flc(sid):
    t = tag(sid); gt = L(f"{J}/cache/b9/gt_{t}_test.pkl"); bp = L(f"{J}/cache/b6/baseline_{t}.pkl")
    ss = [L(f"{J}/cache/b9/STm_{t}_seed{k}.pkl")["test"] for k in (0, 1, 2)]
    r = one_setting(gt, bp["test"], ss); print("flc", t, {k: np.round(np.mean(v), 1) for k, v in r.items()}, flush=True); return t, r
def b12(args):
    src, sid = args; t = tag(sid); gt = L(f"{J}/cache/b9/gt_{t}_test.pkl"); bp = L(f"{J}/cache/b12/{src}_base_{t}.pkl")
    ss = [L(f"{J}/cache/b12/B12{src}_{t}_seed{k}.pkl")["test"] for k in (0, 1, 2) if os.path.exists(f"{J}/cache/b12/B12{src}_{t}_seed{k}.pkl")]
    return (src, t), (one_setting(gt, bp["test"], ss) if ss else None)
def mot(sid):
    gt = load_gt(sorted(glob.glob(f"{J}/data/mot/s{sid}/test/images/*.jpg"))); bp = L(f"{J}/cache/mot/base_s{sid}.pkl")
    ss = [L(f"{J}/cache/mot/EXm_s{sid}_seed{k}.pkl")["test"] for k in (0, 1, 2)]
    return str(sid), one_setting(gt, bp["test"], ss)
def ext(sid):
    man = json.load(open(f"{J}/data/ext/manifest.json")); D = f"{J}/data/ext/{man[str(sid)]['dataset']}/s{sid}"
    gt = load_gt(sorted(glob.glob(f"{D}/test/images/*.jpg"))); bp = L(f"{J}/cache/ext/base_s{sid}.pkl")
    ss = [L(f"{J}/cache/ext/EXm_s{sid}_seed{k}.pkl")["test"] for k in (0, 1, 2)]
    return str(sid), one_setting(gt, bp["test"], ss)
def cityscapes():
    gt = load_gt(sorted(glob.glob(f"{J}/data/cityscapes/test/images/*.png"))); bp = L(f"{J}/cache/cityscapes/base.pkl")
    ss = [L(f"{J}/cache/cityscapes/CSt055_seed{k}.pkl")["test"] for k in (0, 1, 2)]
    return one_setting(gt, bp["test"], ss)
if __name__ == "__main__":
    SC = [s for s, *_ in scenarios()]; out = {}
    with Pool(8) as p: out["flc"] = dict(p.map(flc, SC))
    srcs = [s for s in ("v8s", "coco", "v8m") if os.path.exists(f"{J}/results/b12_{s}.json")]
    with Pool(8) as p: res = p.map(b12, [(s, sid) for s in srcs for sid in SC])
    out["b12"] = {s: {t: r for (ss, t), r in res if ss == s and r} for s in srcs}
    man = json.load(open(f"{J}/data/ext/manifest.json"))
    with Pool(8) as p: out["ext"] = dict(p.map(ext, sorted(int(k) for k in man)))
    mm = json.load(open(f"{J}/data/mot/manifest.json"))
    with Pool(8) as p: out["mot"] = dict(p.map(mot, sorted(int(k) for k in mm)))
    out["cityscapes"] = cityscapes()
    json.dump(out, open(f"{J}/results/b11_fusion.json", "w"), indent=1, default=float)
    def summ(name, d):
        b = np.array([x["base"] for x in d.values()])
        s = f"{name:14s} base {b.mean():5.1f}"
        for k in ("final", "fuse1", "fuse9"):
            v = np.array([np.mean(x[k]) for x in d.values()]); dd = v - b; s += " | %s %5.1f (%d/%d, worst %.1f)" % (k, v.mean(), (dd > 2).sum(), (dd < -2).sum(), dd.min())
        print(s, flush=True)
    summ("23 recordings", out["flc"])
    for s in out["b12"]: summ(s, out["b12"][s])
    summ("MOT fixed", {k: v for k, v in out["mot"].items() if mm[k]["mount"] == "static"}); summ("MOT moving", {k: v for k, v in out["mot"].items() if mm[k]["mount"] == "moving"})
    for ds, mt in (("camel", "tripod"), ("camel", "handheld"), ("birdsai", "aerial")): summ(f"{ds} {mt}", {k: v for k, v in out["ext"].items() if man[k]["dataset"] == ds and man[k]["mount"] == mt})
    c = out["cityscapes"]; print("cityscapes base %.1f final %.1f fuse1 %.1f fuse9 %.1f" % (c["base"], np.mean(c["final"]), np.mean(c["fuse1"]), np.mean(c["fuse9"])))
    print("B11C_DONE")
