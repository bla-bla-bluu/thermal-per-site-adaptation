"""Batch 9a: pseudo-label sets for the new ablations, all at tau = 0.55 (the main
threshold), from cached detections (no new teacher runs):
  t055ne   frames without pseudo-labels are left out of the student's training set (instead of serving as background)
  pers055  static boxes removed: a pseudo-label is dropped if a teacher box (>= 0.55) overlaps it (IoU >= 0.5) in at
           least 80% of the frames t-10..t+10 (the rule of batch 3, now at tau = 0.55)
  ns055    teacher without slicing: the source detector's own full-frame detections (YOLO.predict, conf >= 0.55)
Also pseudo-label statistics on the labelled pool frames (frames, frames with pseudo-labels, boxes per frame, precision,
recall, people in frames that have no pseudo-label and therefore train as background).
Writes data/flc/<tag>/pl_<var>, data/flc/<tag>_<var>.yaml and results/b9_plsets.json."""
import os, json, pickle
import numpy as np
from PIL import Image
os.environ.setdefault("JDOMAIN", "all")
from jcommon import *
import cr_eval as E

def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"

def persistent(D, t, k, W=10, ps=0.8):
    nb = [D[u] for u in range(max(0, t - W), min(len(D), t + W + 1)) if u != t]
    if not nb: return False
    return np.mean([len(x) > 0 and E.iou_mat(D[t][k:k+1], x).max() >= 0.5 for x in nb]) >= ps

def write_set(sid, var, rec, boxes, keep=None):
    t = tag(sid); d = f"{J}/data/flc/{t}/pl_{var}"
    os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
    for i, r in enumerate(rec):
        if keep is not None and not keep[i]: continue
        src = os.path.realpath(f"{sdir(sid, 'pool')}/images/{r['im']}")
        if not os.path.islink(f"{d}/images/{r['im']}"): os.symlink(src, f"{d}/images/{r['im']}")
        W, H = map(float, Image.open(src).size)
        open(f"{d}/labels/{os.path.splitext(r['im'])[0]}.txt", "w").write("".join(
            f"0 {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n" for b in boxes[i]))
    open(f"{J}/data/flc/{t}_{var}.yaml", "w").write(f"path: {d}\ntrain: images\nval: {sdir(sid, 'val')}/images\nnames:\n  0: person\n")

def quality(rec, boxes, labelled, keep=None):
    """Precision/recall of the pseudo-labels on the labelled pool frames that are in the training set."""
    tp = nb = ng = bgp = bgf = nf = nfp = 0
    for i, r in enumerate(rec):
        if keep is not None and not keep[i]: continue
        nf += 1; nfp += int(len(boxes[i]) > 0)
        if os.path.splitext(r["im"])[0] not in labelled: continue
        D = np.asarray(boxes[i]).reshape(-1, 4); G = r["gt"]; nb += len(D); ng += len(G)
        if len(D) == 0 and len(G): bgp += len(G); bgf += 1
        if len(D) and len(G):
            M = E.iou_mat(D, G); used = np.zeros(len(G), bool)
            for k in range(len(D)):        # boxes are in descending confidence order where it matters (sets keep it)
                row = np.where(used, -1, M[k]); j = int(np.argmax(row))
                if row[j] >= 0.5: used[j] = True; tp += 1
    return {"frames": nf, "frames_with_pl": nfp, "boxes_per_frame": nb / max(nf, 1), "P": tp / max(nb, 1), "R": tp / max(ng, 1),
            "people_labelled": ng, "people_in_empty_frames": bgp, "empty_frames_with_people": bgf}

out = {}
for sid, *_ in scenarios():
    t = tag(sid)
    rec = sorted(pickle.load(open(f"{J}/cache/teacher_{t}_pool.pkl", "rb")), key=lambda r: r["im"])
    labelled = {f[:-4] for f in os.listdir(f"{sdir(sid, 'pool')}/labels") if f.endswith(".txt")}
    order = [np.argsort(-r["conf"], kind="stable") for r in rec]
    D55 = [r["det"][o][r["conf"][o] >= 0.55] for r, o in zip(rec, order)]
    # frames without pseudo-labels left out
    keep = [len(d) > 0 for d in D55]
    write_set(sid, "t055ne", rec, D55, keep)
    # static boxes removed
    P55 = [np.array([D55[i][k] for k in range(len(D55[i])) if not persistent(D55, i, k)]).reshape(-1, 4) for i in range(len(D55))]
    write_set(sid, "pers055", rec, P55)
    # teacher without slicing: the cached full-frame predictions of the source detector (aligned with the sorted pool)
    bp = pickle.load(open(f"{J}/cache/Baseline_{t}_pool_predict.pkl", "rb")); assert len(bp) == len(rec), (sid, len(bp), len(rec))
    NS = [d[np.argsort(-c, kind="stable")][np.sort(c)[::-1] >= 0.55] for d, c in bp]
    write_set(sid, "ns055", rec, NS)
    out[t] = {"t055": quality(rec, D55, labelled), "t055ne": quality(rec, D55, labelled, keep),
              "pers055": quality(rec, P55, labelled), "ns055": quality(rec, NS, labelled)}
    print(t, {k: (v["frames"], v["frames_with_pl"], round(v["P"], 2), round(v["R"], 2), v["people_in_empty_frames"]) for k, v in out[t].items()}, flush=True)
json.dump(out, open(f"{J}/results/b9_plsets.json", "w"), indent=1)
print("B9_PREP_DONE")
