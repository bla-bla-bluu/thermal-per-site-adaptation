"""Journal prep: audit the 20 FLC-Rural scenarios as UDA target domains.
Per scenario: frame range in the rendered 8-bit copies (yolo_ir_oracle/train for s01-s14, yolo_flc for s15-s20),
temporal continuity (mean abs diff of consecutive frames vs random pairs), people per frame, empty frames,
median person box (px). Also checks that ir_dataset/train/NNNNNN.png == FLC-Rural s.._ stems in manifest order."""
import csv, os, random, collections
import numpy as np
from PIL import Image
B = "/path/to/domain"; F = "/path/to/camo/FLC-Rural"
rows = list(csv.DictReader(open(f"{F}/metadata/scenario_manifest.csv")))
scen = collections.OrderedDict()
for r in rows:
    scen.setdefault((int(r["scenario_id"]), r["scenario_name"], r["split"]), []).append(r["image_stem"])
# mapping check: k-th train stem in manifest order <-> ir_dataset/train/{k:06d}.png
tr = [r["image_stem"] for r in rows if r["split"] == "train"]; te = [r["image_stem"] for r in rows if r["split"] == "test"]
for k in (0, 500, 4000, 9581):
    a = np.asarray(Image.open(f"{F}/images/train/ir/{tr[k]}.png")); b = np.asarray(Image.open(f"/path/to/camo/ir_dataset/train/images/{k+1:06d}.png"))
    print("map train", k + 1, tr[k], "identical" if a.shape == b.shape and (a == b).all() else "DIFFERENT")
for k in (0, 2000, 4172):
    a = np.asarray(Image.open(f"{F}/images/test/ir/{te[k]}.png")); b = np.asarray(Image.open(f"/path/to/camo/ir_dataset/test/images/{k+1:06d}.png"))
    print("map test ", k + 1, te[k], "identical" if a.shape == b.shape and (a == b).all() else "DIFFERENT")
random.seed(0); tidx = 0; eidx = 0
print(f"{'sid':>3} {'name':34s} {'split':5s} {'frames':>6} {'idx range':>13} {'cons/rand diff':>15} {'ppl/fr':>6} {'empty':>5} {'median box':>10} {'animals':>7}")
for (sid, name, split), stems in scen.items():
    n = len(stems)
    if split == "train":
        lo = tidx + 1; tidx += n; d = f"{B}/yolo_ir_oracle/train"; ids = [f"{k:06d}" for k in range(lo, lo + n)]
    else:
        lo = eidx + 1; eidx += n; sub = [x for x in os.listdir(f"{B}/yolo_flc") if x.startswith(f"s{sid}_")][0]
        d = f"{B}/yolo_flc/{sub}"; ids = [f"{k:06d}" for k in range(lo, lo + n)]
    load = lambda i: np.asarray(Image.open(f"{d}/images/{ids[i]}.jpg").convert("L"), dtype=np.float32)
    pick = random.sample(range(n - 1), min(40, n - 1))
    cons = np.mean([np.abs(load(i) - load(i + 1)).mean() for i in pick])
    rnd = np.mean([np.abs(load(i) - load(random.randrange(n))).mean() for i in pick])
    ppl, empty, wh, animals = [], 0, [], 0
    for i in ids:
        L = [l.split() for l in open(f"/path/to/camo/ir_dataset/{split}/labels/{i}.txt") if l.strip()]
        P = [q for q in L if q[0] == "0"]; animals += sum(q[0] == "1" for q in L)
        ppl.append(len(P)); empty += len(P) == 0
        wh += [(float(q[3]) * 680, float(q[4]) * 480) for q in P]
    wh = np.array(wh) if wh else np.zeros((0, 2))
    mb = f"{np.median(wh[:,0]):.0f}x{np.median(wh[:,1]):.0f}" if len(wh) else "-"
    print(f"{sid:>3} {name:34s} {split:5s} {n:>6} {lo:>6}-{lo+n-1:<6} {cons:6.1f}/{rnd:6.1f} {np.mean(ppl):6.2f} {empty:>5} {mb:>10} {animals:>7}", flush=True)
