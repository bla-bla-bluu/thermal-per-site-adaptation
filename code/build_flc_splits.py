"""Build per-scenario split folders (symlinked 8-bit images + person-only labels) and YOLO data yamls."""
import os, json
from jcommon import *
manifest = {}
for sid, name, split, imgs, labs in scenarios():
    n = len(imgs); bnd = split_bounds(n); manifest[sid] = {"name": name, "flc_split": split, "frames": n}
    for part, (lo, hi) in bnd.items():
        d = sdir(sid, part); os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
        npers = 0
        for t in range(lo, hi):
            stem = f"{t:05d}"                       # temporal index within the scenario
            dst = f"{d}/images/{stem}.jpg"
            if not os.path.islink(dst):
                os.symlink(imgs[t], dst)
            P = [l.split() for l in open(labs[t]) if l.strip()]
            P = [q for q in P if q[0] == "0"]; npers += len(P)
            open(f"{d}/labels/{stem}.txt", "w").write("".join("0 " + " ".join(q[1:5]) + "\n" for q in P))
        open(f"{J}/data/flc/s{sid:02d}_{part}.yaml", "w").write(f"path: {d}\ntrain: images\nval: images\nnames:\n  0: person\n")
        manifest[sid][part] = {"frames": hi - lo, "people": npers, "range": [lo, hi]}
    print(sid, name, {p: manifest[sid][p]["frames"] for p in SPLITS}, flush=True)
json.dump(manifest, open(f"{J}/data/flc/manifest.json", "w"), indent=1)
print("SPLITS_DONE")
