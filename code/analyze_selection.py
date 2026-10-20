"""Batch-2a analysis: how well do label-free signals pick a good checkpoint? For each scene: test AP50 of the
final epoch, of the epoch chosen by labelled validation (val mAP50), by each label-free signal on the gap frames,
and of the best epoch on the test set itself (the unattainable optimum). Also mean Spearman correlation per signal."""
import json, glob, collections
import numpy as np
J = "/path/to/domain/journal"
runs = collections.defaultdict(list)
for f in glob.glob(f"{J}/results/selection_w*.jsonl"):
    for l in open(f):
        r = json.loads(l); runs[r["sid"]].append(r)
def spearman(a, b):
    a, b = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
SIG = {"val": lambda r: r["val_mAP50"] or 0, "agree": lambda r: r["agree_AP50"], "flip": lambda r: r["flip_AP50"],
       "temporal": lambda r: r["temporal"], "agree+flip": lambda r: r["agree_AP50"] + r["flip_AP50"]}
rows, rho = [], collections.defaultdict(list)
print(f"{'sid':>3} {'final':>6} {'ep10':>6} {'ep20':>6} " + " ".join(f"{k:>10}" for k in SIG) + f" {'test-opt':>8}")
for sid in sorted(runs):
    R = sorted(runs[sid], key=lambda r: r["epoch"])
    if len(R) < 50: print(sid, "incomplete", len(R)); continue
    t = np.array([r["test_AP50"] for r in R]) * 100
    row = {"final": t[-1], "ep10": t[9], "ep20": t[19], "test-opt": t.max()}
    for k, f in SIG.items():
        s = np.array([f(r) for r in R]); row[k] = t[int(np.argmax(s))]; rho[k].append(spearman(s, t))
    rows.append(row)
    print(f"{sid:>3} {row['final']:6.1f} {row['ep10']:6.1f} {row['ep20']:6.1f} " + " ".join(f"{row[k]:10.1f}" for k in SIG) + f" {row['test-opt']:8.1f}")
if rows:
    m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    print(f"{'mean':>3} {m['final']:6.1f} {m['ep10']:6.1f} {m['ep20']:6.1f} " + " ".join(f"{m[k]:10.1f}" for k in SIG) + f" {m['test-opt']:8.1f}")
    print("mean Spearman(signal, test AP50) over epochs:", {k: round(float(np.nanmean(v)), 2) for k, v in rho.items()})
    print("scenes:", len(rows))
