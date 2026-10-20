"""Batch-1 report: per-scenario baseline (std / SAHI), pseudo-label quality of the SAHI teacher at several thresholds,
and (when available) the self-trained student and the supervised in-domain reference. AP50 in %."""
import json, os, collections
J = "/path/to/domain/journal"
rows = [json.loads(l) for l in open(f"{J}/results/teacher_baseline.jsonl")]
q = {r["sid"]: r["q"] for r in rows if r.get("kind") == "pseudo_label_quality"}
res = collections.defaultdict(dict); names = {}
for r in rows:
    if "model" in r: res[r["sid"]][(r["model"], r["mode"])] = r; names[r["sid"]] = r["name"]
if os.path.exists(f"{J}/results/flc_models.jsonl"):
    for l in open(f"{J}/results/flc_models.jsonl"):
        r = json.loads(l); res[r["sid"]][(r["model"], r["mode"])] = r
f = lambda sid, m, mode, k="AP50": f"{100 * res[sid][(m, mode)][k]:5.1f}" if (m, mode) in res[sid] else "   - "
print(f"{'sid':>3} {'scenario':32s} {'Base':>5} {'B+SAHI':>6} {'ST':>5} {'ST+S':>5} {'STval':>5} {'STsrc':>5} {'Src+S':>5} {'SUP':>5} | {'PL/fr .45':>9} {'P':>5} {'R':>5} | {'PL/fr .25':>9} {'P':>5} {'R':>5}")
for sid in sorted(res):
    a = q.get(sid, {}).get("0.45", {}); c = q.get(sid, {}).get("0.25", {})
    print(f"{sid:>3} {names[sid][:32]:32s} {f(sid,'Baseline','std')} {f(sid,'Baseline','sahi'):>6} {f(sid,'ST-last','std')} {f(sid,'ST-last','sahi')} "
          f"{f(sid,'ST-valbest','std')} {f(sid,'STsrc-last','std')} {f(sid,'STsrc-last','sahi')} {f(sid,'SUP-last','std')} | {a.get('boxes_per_frame', 0):9.2f} {100*a.get('P', 0):5.1f} {100*a.get('R', 0):5.1f} | "
          f"{c.get('boxes_per_frame', 0):9.2f} {100*c.get('P', 0):5.1f} {100*c.get('R', 0):5.1f}")

import numpy as np
cols = [("Baseline", "std"), ("Baseline", "sahi"), ("ST-last", "std"), ("ST-last", "sahi"), ("ST-valbest", "std"), ("STsrc-last", "std"), ("STsrc-last", "sahi"), ("SUP-last", "std")]
full = [sid for sid in res if all(c in res[sid] for c in cols)]
if full:
    print("mean over", len(full), "scenes:", {f"{m}/{md}": round(float(np.mean([100 * res[s][(m, md)]["AP50"] for s in full])), 1) for m, md in cols})
    for m in ("ST-last", "STsrc-last"):
        d = [100 * (res[s][(m, "std")]["AP50"] - res[s][("Baseline", "std")]["AP50"]) for s in full]
        print(m, "vs baseline: helps", sum(x > 2 for x in d), "hurts", sum(x < -2 for x in d), "worst %.1f" % min(d))
