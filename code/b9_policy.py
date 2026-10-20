"""Batch 9f: the policy, selection and metric analyses of the revised paper, from the batch-9 caches
(no new predictions). For every recording and seed of the main tau=0.55 students (STm):
  * test AP50, AP75, AP50:95 (101-point AP after greedy matching at each IoU in 0.50:0.05:0.95) and P/R at conf 0.25 of
    every candidate: the source detector theta0 and the 50 epochs
  * few-label guard: n labelled validation frames (n = 3, 5, 10, 20, 40 or the whole window), drawn at random or as one
    contiguous clip starting at a random frame, 50 draws each (one draw when n covers the window); the candidate with the
    best AP50 on them is deployed (ties: theta0, then the earlier epoch), with theta0 among the candidates ("+base") or
    not. If the n frames contain no person, the candidates cannot be compared and the guard keeps theta0 ("+base") or
    the final epoch. Draws: numpy default_rng([sid, seed, n, mode]). Per draw we keep the chosen candidate, its AP50 on
    the n frames and on the test window, the same for the final epoch, and the mean over all candidates (optimism).
  * label-free checkpoint scores on the gap frames: agreement with the teacher's pseudo-labels (tau = 0.55 and 0.45),
    flip consistency, temporal consistency; fixed epoch 10; the whole validation window; the test-best epoch (oracle);
    Spearman correlation of each score with test AP50 over the epochs
  * teacher advantage (teacher AP50 minus theta0 AP50) on the test window, on the whole validation window and on 10
    random validation frames (same draws as the guard, seed 0)
Variants STsrc, STne, STns, STpe: final epoch, 10-frame guard with and without theta0, oracle. AdaBN and the supervised
reference: test metrics. Pooled AP50 over the test frames of all recordings. Writes results/b9_policy.json."""
import os, json, pickle
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
SC = [s for s, *_ in scenarios()]; IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
NS = (3, 5, 10, 20, 40, 0); VARS = ("STsrc", "STne", "STns", "STpe"); ND = 50
def L(f): return pickle.load(open(f, "rb"))

def fmatch(gt, det, thr=0.5):
    """Per-frame greedy matching: [(conf, tp)] per frame, and the number of people per frame."""
    out = []
    for g, (d, c) in zip(gt, det):
        dd, _ = E.match([{"gt": g, "det": d, "conf": c, "gain": 1.0}], thr); out.append((dd[:, 0], dd[:, 1]))
    return out, np.array([len(g) for g in gt])
def ap_idx(fm, ngt, idx):
    conf = np.concatenate([fm[i][0] for i in idx]); tp = np.concatenate([fm[i][1] for i in idx])
    o = np.argsort(-conf, kind="stable"); t = tp[o]; ctp = np.cumsum(t); cfp = np.cumsum(1 - t)
    return 100 * E.ap101(ctp / max(int(ngt[idx].sum()), 1), ctp / np.maximum(ctp + cfp, 1e-9))
def pr_at(fm, ngt, op=0.25):
    conf = np.concatenate([f[0] for f in fm]); tp = np.concatenate([f[1] for f in fm]); k = conf >= op
    P = 100 * tp[k].sum() / max(k.sum(), 1); R = 100 * tp[k].sum() / max(ngt.sum(), 1); return P, R
def metrics(gt, det):
    out, aps = {}, []
    for thr in IOUS:
        fm, ngt = fmatch(gt, det, thr); aps.append(ap_idx(fm, ngt, np.arange(len(gt))))
        if thr == 0.5: out["P"], out["R"] = pr_at(fm, ngt)
    out.update(AP50=aps[0], AP75=aps[5], AP5095=float(np.mean(aps))); return out
def temporal(det, thr=0.25):
    hit = tot = 0
    for (d0, c0), (d1, c1) in zip(det[:-1], det[1:]):
        A, B_ = d0[c0 >= thr], d1[c1 >= thr]
        for X, Y in ((A, B_), (B_, A)):
            if len(X) == 0: continue
            tot += len(X); hit += int((E.iou_mat(X, Y).max(1) >= 0.3).sum()) if len(Y) else 0
    return hit / tot if tot else 0.0
def draws(sid, seed, n, mode, nv):
    """Index sets of the labelled frames: 50 draws, or the whole window once."""
    if n == 0 or n >= nv: return [np.arange(nv)]
    rng = np.random.default_rng([sid, seed, n, 0 if mode == "random" else 1])
    if mode == "random": return [rng.choice(nv, n, replace=False) for _ in range(ND)]
    return [np.arange(s0, s0 + n) for s0 in rng.integers(0, nv - n + 1, ND)]
def guard(fmv, ngtv, eps, idx, with_base):
    """Chosen candidate (0 = theta0, e = epoch) and the scores on the labelled frames."""
    if ngtv[idx].sum() == 0: return (0 if with_base else eps[-1]), None
    sc = {e: ap_idx(fmv[e], ngtv, idx) for e in eps}; best = max(eps, key=lambda e: (sc[e], -e))
    if with_base and ap_idx(fmv[0], ngtv, idx) >= sc[best]: return 0, sc
    return best, sc

def one(sid):
    t = tag(sid); gv, gt = L(f"{J}/cache/b9/gt_{t}_val.pkl"), L(f"{J}/cache/b9/gt_{t}_test.pkl"); nv = len(gv)
    bp = L(f"{J}/cache/b6/baseline_{t}.pkl"); r = {"nv": nv, "nt": len(gt), "people_test": int(sum(len(g) for g in gt))}
    fm0v, ngtv = fmatch(gv, bp["val"]); fm0t, ngtt = fmatch(gt, bp["test"])
    r["base"] = metrics(gt, bp["test"]); r["base_val"] = ap_idx(fm0v, ngtv, np.arange(nv))
    # teacher advantage
    tt = L(f"{J}/cache/Baseline_{t}_test_sahi.pkl"); tv = L(f"{J}/cache/b9/tval_{t}.pkl")
    fmtv, _ = fmatch(gv, [(x["det"], x["conf"]) for x in tv])
    tadv10 = [ap_idx(fmtv, ngtv, i) - ap_idx(fm0v, ngtv, i) for i in draws(sid, 0, 10, "random", nv) if ngtv[i].sum()]
    r["teacher"] = {"test": metrics([x["gt"] for x in tt], [(x["det"], x["conf"]) for x in tt])["AP50"] - r["base"]["AP50"],
                    "val_all": ap_idx(fmtv, ngtv, np.arange(nv)) - r["base_val"], "val10": float(np.mean(tadv10)) if tadv10 else 0.0}
    # teacher pseudo-labels on the gap frames, for the agreement score
    tg = sorted(L(f"{J}/cache/teacher_{t}_gap.pkl"), key=lambda x: x["im"])
    pg = {tau: [x["det"][x["conf"] >= tau] for x in tg] for tau in (0.45, 0.55)}
    M = {"final": [], "opt": [], "pol": {}, "perdraw": {}, "opt_stats": {},
         "lf": {k: {"ap": [], "rho": [], "ep": []} for k in ("agr", "agr45", "flip", "tmp", "ep10", "valall")}, "cand_test": [], "sel10": []}
    for seed in (0, 1, 2):
        c = L(f"{J}/cache/b9/STm_{t}_seed{seed}.pkl"); eps = sorted(c["test"])
        test = {e: metrics(gt, c["test"][e]) for e in eps}; test[0] = r["base"]
        fmv = {e: fmatch(gv, c["val"][e])[0] for e in eps}; fmv[0] = fm0v
        a50 = np.array([test[e]["AP50"] for e in eps]); M["cand_test"].append([test[e]["AP50"] for e in [0] + eps])
        M["final"].append(test[eps[-1]]); M["opt"].append(max(test.values(), key=lambda m: m["AP50"]))
        for mode in ("random", "clip"):
            for n in NS:
                if mode == "clip" and n == 0: continue
                for wb in (False, True):
                    key = f"{'pick' if mode == 'random' else 'clip'}{n if n else 'all'}{'+base' if wb else ''}"
                    ch, sv, st_, fv, mv = [], [], [], [], []
                    for idx in draws(sid, seed, n, mode, nv):
                        e, sc = guard(fmv, ngtv, eps, idx, wb); ch.append(e)
                        if sc is not None and not wb:
                            sv.append(sc[e]); st_.append(test[e]["AP50"]); fv.append(sc[eps[-1]]); mv.append(np.mean(list(sc.values())))
                    vals = {k: [test[e][k] for e in ch] for k in ("AP50", "AP75", "AP5095", "P", "R")}
                    d = M["pol"].setdefault(key, {k: [] for k in ("AP50", "AP75", "AP5095", "P", "R", "sd_draw", "base_share", "no_person")})
                    for k in vals: d[k].append(float(np.mean(vals[k])))
                    d["sd_draw"].append(float(np.std(vals["AP50"]))); d["base_share"].append(float(np.mean([e == 0 for e in ch])))
                    d["no_person"].append(float(np.mean([ngtv[i].sum() == 0 for i in draws(sid, seed, n, mode, nv)])))
                    if mode == "random" and not wb and sv:      # selection optimism: score on the n frames minus test AP50
                        o = M["opt_stats"].setdefault(str(n if n else "all"), {"sel_val": [], "sel_test": [], "final_val": [], "final_test": [], "mean_val": [], "mean_test": []})
                        o["sel_val"].append(float(np.mean(sv))); o["sel_test"].append(float(np.mean(st_)))
                        o["final_val"].append(float(np.mean(fv))); o["final_test"].append(test[eps[-1]]["AP50"])
                        o["mean_val"].append(float(np.mean(mv))); o["mean_test"].append(float(a50.mean()))
                    M["perdraw"].setdefault(key, []).append([test[e]["AP50"] for e in ch])
                    if key == "pick10+base": M["sel10"].append(ch)
        # label-free scores on the gap frames
        S = {"agr": [], "agr45": [], "flip": [], "tmp": []}
        for e in eps:
            g = c["gap"][e]
            S["agr"].append(ap_idx(*fmatch(pg[0.55], g), np.arange(len(g)))); S["agr45"].append(ap_idx(*fmatch(pg[0.45], g), np.arange(len(g))))
            ref = [d[cc >= 0.25] for d, cc in g]; S["flip"].append(ap_idx(*fmatch(ref, c["gapflip"][e]), np.arange(len(g))))
            S["tmp"].append(temporal(g))
        S["valall"] = [ap_idx(fmv[e], ngtv, np.arange(nv)) for e in eps]
        for k, v in S.items():
            i = int(np.argmax(v)); M["lf"][k]["ap"].append(float(a50[i])); M["lf"][k]["ep"].append(eps[i])
            rho = spearmanr(v, a50).correlation if np.std(v) > 0 and np.std(a50) > 0 else np.nan; M["lf"][k]["rho"].append(float(rho))
        M["lf"]["ep10"]["ap"].append(float(a50[9]))
    r["main"] = M
    # variants: final epoch, 10-frame guard with and without theta0, oracle
    r["var"] = {}
    for pre in VARS:
        V = {"final": [], "pick10": [], "pick10+base": [], "opt": []}
        for seed in (0, 1, 2):
            f = f"{J}/cache/b9/{pre}_{t}_seed{seed}.pkl"
            if not os.path.exists(f): continue
            c = L(f); eps = sorted(c["test"]); fmt_ = {e: fmatch(gt, c["test"][e]) for e in eps}
            ta = {e: ap_idx(fmt_[e][0], fmt_[e][1], np.arange(len(gt))) for e in eps}; ta[0] = r["base"]["AP50"]
            fmv = {e: fmatch(gv, c["val"][e])[0] for e in eps}; fmv[0] = fm0v
            V["final"].append(ta[eps[-1]]); V["opt"].append(max(ta.values()))
            for wb in (False, True):
                V["pick10+base" if wb else "pick10"].append(float(np.mean([ta[guard(fmv, ngtv, eps, i, wb)[0]] for i in draws(sid, seed, 10, "random", nv)])))
        r["var"][pre] = V
    a = L(f"{J}/cache/b9/adabn_{t}.pkl"); r["adabn"] = metrics(gt, a["test"])
    s = L(f"{J}/cache/b9/sup_{t}.pkl"); r["sup"] = [metrics(gt, s[k]) for k in (0, 1, 2)]
    print("done", t, flush=True)
    return t, r

if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "12"))) as pool: R = dict(pool.map(one, SC))
    # pooled AP50 over the test frames of all recordings (frames of different recordings simply concatenated)
    FM = {}
    for sid in SC:
        t = tag(sid); gt = L(f"{J}/cache/b9/gt_{t}_test.pkl"); bp = L(f"{J}/cache/b6/baseline_{t}.pkl")
        FM[t] = {"base": fmatch(gt, bp["test"]), "sup": [fmatch(gt, L(f"{J}/cache/b9/sup_{t}.pkl")[k]) for k in (0, 1, 2)]}
        FM[t]["st"] = []
        for k in (0, 1, 2):             # only the epochs that a policy deploys: the final one and the 10-frame guard's picks
            need = {50} | {e for e in R[t]["main"]["sel10"][k] if e}; x = L(f"{J}/cache/b9/STm_{t}_seed{k}.pkl")["test"]
            FM[t]["st"].append({e: fmatch(gt, x[e])[0] for e in need})
    def pooled(pick):
        fm, ngt = [], []
        for t in FM:
            f, n = pick(t); fm += f; ngt.append(n)
        ngt = np.concatenate(ngt); return ap_idx(fm, ngt, np.arange(len(fm)))
    P = {"base": pooled(lambda t: FM[t]["base"]),
         "sup": [pooled(lambda t: FM[t]["sup"][k]) for k in (0, 1, 2)],
         "final": [pooled(lambda t: (FM[t]["st"][k][50], FM[t]["base"][1])) for k in (0, 1, 2)]}
    g = []
    for k in (0, 1, 2):
        for d in range(ND):
            def pk(t):
                ch = R[t]["main"]["sel10"][k]; e = ch[d] if d < len(ch) else ch[-1]
                return (FM[t]["base"][0] if e == 0 else FM[t]["st"][k][e]), FM[t]["base"][1]
            g.append(pooled(pk))
    P["pick10+base"] = g
    json.dump({"per": R, "pooled": P}, open(f"{J}/results/b9_policy.json", "w"), indent=1, default=float)
    print("B9_POLICY_DONE")
