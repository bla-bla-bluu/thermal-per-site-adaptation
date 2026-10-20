"""Batch 11b-2: the multi-variant guard, with the variant sets fixed in advance and no tuning on the test windows.
Candidates: the source detector theta0 and the 50 epochs of every trajectory in a variant set S, chosen by AP50 on n labelled
validation frames (the paper's guard, ties to theta0, then to the earlier trajectory in the order main, empty-out, static-out,
source-init, then to the earlier epoch). Variant sets: M = {main}, MN = {main, empty-out}, MNS = {+ source-init},
MNP = {+ static-out}, ALL = all four; n = 5, 10, 20 labelled frames (50 random draws, same draws as the paper).
  (1) the 23 FLC-Rural/forest recordings (caches cache/b9/STm|STne|STpe|STsrc_<tag>_seed<k>.pkl)
  (2) CAMEL and BIRDSAI (cache/ext/EXm|EXne_s<id>_seed<k>.pkl; sets M and MN)
  (3) Cityscapes -> Foggy (cache/cityscapes/CSt055|CSt055ne_seed<k>.pkl; sets M and MN; guard frames are the 100 labelled images)
Writes results/b11_subsets.json."""
import os, sys, json, glob, pickle
import numpy as np
from multiprocessing import Pool
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
from b9_policy import fmatch, ap_idx, draws, L
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
ORDER = {"0": 0, "STm": 1, "STne": 2, "STpe": 3, "STsrc": 4}
SETS = {"M": ["STm"], "MN": ["STm", "STne"], "MNS": ["STm", "STne", "STsrc"], "MNP": ["STm", "STne", "STpe"], "ALL": ["STm", "STne", "STpe", "STsrc"]}
def multi(gv_fm, ngtv, base_val_fm, TA, FVd, base, idxs, sets):
    """Mean test AP50 over the draws for each variant set (TA/FVd keyed (pre, epoch); theta0 = ('0', 0))."""
    out = {s: [] for s in sets}; keys_all = sorted(TA, key=lambda k: (ORDER[k[0]], k[1]))
    for idx in idxs:
        if ngtv[idx].sum() == 0:
            for s in sets: out[s].append(base)
            continue
        sc = {k: (ap_idx(base_val_fm, ngtv, idx) if k[0] == "0" else ap_idx(FVd[k], ngtv, idx)) for k in keys_all}
        for s, pres in sets.items():
            ks = [k for k in keys_all if k[0] == "0" or k[0] in pres]; best = max(ks, key=lambda k: (sc[k], -ORDER[k[0]], -k[1])); out[s].append(TA[best])
    return {s: float(np.mean(v)) for s, v in out.items()}
def flc(sid):
    t = tag(sid); gv, gt = L(f"{J}/cache/b9/gt_{t}_val.pkl"), L(f"{J}/cache/b9/gt_{t}_test.pkl"); nv, nt = len(gv), len(gt)
    bp = L(f"{J}/cache/b6/baseline_{t}.pkl"); fm0v, ngtv = fmatch(gv, bp["val"]); base = ap_idx(*fmatch(gt, bp["test"]), np.arange(nt)); R = {"base": base, "seeds": []}
    for seed in (0, 1, 2):
        TA = {("0", 0): base}; FVd = {}
        for pre in ("STm", "STne", "STpe", "STsrc"):
            cf = f"{J}/cache/b9/{pre}_{t}_seed{seed}.pkl"
            if not os.path.exists(cf): continue
            c = L(cf)
            for e in sorted(c["test"]): TA[(pre, e)] = ap_idx(*fmatch(gt, c["test"][e]), np.arange(nt)); FVd[(pre, e)] = fmatch(gv, c["val"][e])[0]
        o = {}
        for n in (5, 10, 20): o[f"n{n}"] = multi(None, ngtv, fm0v, TA, FVd, base, draws(sid, seed, n, "random", nv), SETS)
        R["seeds"].append(o)
    print("flc", t, round(base, 1), {s: round(float(np.mean([x["n10"][s] for x in R["seeds"]])), 1) for s in SETS}, flush=True); return t, R
def ext(sid):
    J2 = J; man = json.load(open(f"{J}/data/ext/manifest.json")); D = f"{J}/data/ext/{man[str(sid)]['dataset']}/s{sid}"
    from selection_study_helpers import load_gt
    gv = load_gt(sorted(glob.glob(f"{D}/val/images/*.jpg"))); gt = load_gt(sorted(glob.glob(f"{D}/test/images/*.jpg"))); nv, nt = len(gv), len(gt)
    b = L(f"{J}/cache/ext/base_s{sid}.pkl"); fm0v, ngtv = fmatch(gv, b["val"]); base = ap_idx(*fmatch(gt, b["test"]), np.arange(nt)); R = {"base": base, "seeds": []}
    for seed in (0, 1, 2):
        TA = {("0", 0): base}; FVd = {}
        for pre, nm in (("STm", "EXm"), ("STne", "EXne")):
            cf = f"{J}/cache/ext/{nm}_s{sid}_seed{seed}.pkl"
            if not os.path.exists(cf): continue
            c = L(cf)
            for e in sorted(c["test"]): TA[(pre, e)] = ap_idx(*fmatch(gt, c["test"][e]), np.arange(nt)); FVd[(pre, e)] = fmatch(gv, c["val"][e])[0]
        R["seeds"].append({"n10": multi(None, ngtv, fm0v, TA, FVd, base, draws(sid, seed, 10, "random", nv), {"M": ["STm"], "MN": ["STm", "STne"]})})
    return str(sid), R
def cityscapes():
    from selection_study_helpers import load_gt
    D = f"{J}/data/cityscapes"; C = f"{J}/cache/cityscapes"
    gv = load_gt(sorted(glob.glob(f"{D}/guard/images/*.png"))); gt = load_gt(sorted(glob.glob(f"{D}/test/images/*.png"))); nv, nt = len(gv), len(gt)
    b = L(f"{C}/base.pkl"); fm0v, ngtv = fmatch(gv, b["guard"]); base = ap_idx(*fmatch(gt, b["test"]), np.arange(nt)); R = {"base": base, "seeds": []}
    for seed in (0, 1, 2):
        TA = {("0", 0): base}; FVd = {}
        for pre, nm in (("STm", "CSt055"), ("STne", "CSt055ne")):
            c = L(f"{C}/{nm}_seed{seed}.pkl")
            for e in sorted(c["test"]): TA[(pre, e)] = ap_idx(*fmatch(gt, c["test"][e]), np.arange(nt)); FVd[(pre, e)] = fmatch(gv, c["guard"][e])[0]
        rng = np.random.default_rng([16, seed]); idxs = [rng.choice(nv, 10, replace=False) for _ in range(50)]
        R["seeds"].append({"n10": multi(None, ngtv, fm0v, TA, FVd, base, idxs, {"M": ["STm"], "MN": ["STm", "STne"]})})
    return R
if __name__ == "__main__":
    SC = [s for s, *_ in scenarios()]; out = {}
    with Pool(8) as p: out["flc"] = dict(p.map(flc, SC))
    man = json.load(open(f"{J}/data/ext/manifest.json"))
    with Pool(8) as p: out["ext"] = dict(p.map(ext, sorted(int(k) for k in man)))
    out["cityscapes"] = cityscapes()
    json.dump(out, open(f"{J}/results/b11_subsets.json", "w"), indent=1, default=float)
    T = list(out["flc"]); b = np.array([out["flc"][t]["base"] for t in T])
    for n in (5, 10, 20):
        for s in SETS:
            v = np.array([np.mean([x[f"n{n}"][s] for x in out["flc"][t]["seeds"]]) for t in T]); d = v - b
            print("n=%2d %-4s mean %.1f helps %d hurts %d worst %.1f" % (n, s, v.mean(), (d > 2).sum(), (d < -2).sum(), d.min()))
    for grp, ks in (("camel", [k for k in out["ext"] if man[k]["dataset"] == "camel"]), ("birdsai", [k for k in out["ext"] if man[k]["dataset"] == "birdsai"])):
        for s in ("M", "MN"): print(grp, s, round(float(np.mean([np.mean([x["n10"][s] for x in out["ext"][k]["seeds"]]) for k in ks])), 1))
    print("cityscapes", {s: round(float(np.mean([x["n10"][s] for x in out["cityscapes"]["seeds"]])), 1) for s in ("M", "MN")}, "base", round(out["cityscapes"]["base"], 1))
    print("B11B_DONE")
