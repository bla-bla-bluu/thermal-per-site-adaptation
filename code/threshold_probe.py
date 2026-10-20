"""Probe label-free adaptive thresholds on the cached teacher scores (pool frames) against the hidden pseudo-label
precision/recall: Otsu on scores>=0.1 and a 2-component GMM in logit space on scores>=0.1."""
import pickle, numpy as np
from jcommon import *
import cr_eval as E
def otsu(x, bins=90):
    h, e = np.histogram(x, bins=bins, range=(0.1, 1.0)); c = (e[:-1] + e[1:]) / 2; best, thr = -1, 0.45
    for k in range(1, bins):
        w0, w1 = h[:k].sum(), h[k:].sum()
        if w0 == 0 or w1 == 0: continue
        m0, m1 = (h[:k] * c[:k]).sum() / w0, (h[k:] * c[k:]).sum() / w1; v = w0 * w1 * (m0 - m1) ** 2
        if v > best: best, thr = v, e[k]
    return float(thr)
def gmm_logit(x):
    z = np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
    mu = np.percentile(z, [30, 95]).astype(float); sd = np.array([z.std(), z.std()]) + 1e-3; w = np.array([0.7, 0.3])
    for _ in range(300):
        p = w * np.exp(-0.5 * ((z[:, None] - mu) / sd) ** 2) / sd; r = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
        w = r.mean(0); mu = (r * z[:, None]).sum(0) / r.sum(0); sd = np.sqrt((r * (z[:, None] - mu) ** 2).sum(0) / r.sum(0)) + 1e-3
    hi = int(np.argmax(mu)); g = np.linspace(-4, 4, 801); p = w * np.exp(-0.5 * ((g[:, None] - mu) / sd) ** 2) / sd
    post = p[:, hi] / p.sum(1); idx = np.where((post >= 0.5) & (g >= mu.min()))[0]
    return float(1 / (1 + np.exp(-g[idx[0]]))) if len(idx) else 0.45
def quality(rec, tau):
    tp = nb = ng = 0
    for r in rec:
        keep = r["conf"] >= tau; D = r["det"][keep]; G = r["gt"]; nb += len(D); ng += len(G)
        M = E.iou_mat(D, G); used = np.zeros(len(G), bool)
        for k in np.argsort(-r["conf"][keep]):
            if len(G) == 0: break
            row = np.where(used, -1, M[k]); j = int(np.argmax(row))
            if row[j] >= 0.5: used[j] = True; tp += 1
    return tp / max(nb, 1), tp / max(ng, 1)
print(f"{'sid':>3} {'otsu':>5} {'P/R@otsu':>11} {'gmmL':>5} {'P/R@gmmL':>11} | P/R at .25 .35 .45 .55 .65")
for sid, name, split, imgs, labs in scenarios():
    rec = pickle.load(open(f"{J}/cache/teacher_s{sid:02d}_pool.pkl", "rb"))
    x = np.concatenate([r["conf"] for r in rec]); x = x[x >= 0.1]
    to, tg = otsu(x), gmm_logit(x)
    po, ro = quality(rec, to); pg, rg = quality(rec, tg)
    grid = " ".join(f"{100*p:3.0f}/{100*r:<3.0f}" for p, r in (quality(rec, t) for t in (0.25, 0.35, 0.45, 0.55, 0.65)))
    print(f"{sid:>3} {to:5.2f} {100*po:5.1f}/{100*ro:<5.1f} {tg:5.2f} {100*pg:5.1f}/{100*rg:<5.1f} | {grid}", flush=True)
