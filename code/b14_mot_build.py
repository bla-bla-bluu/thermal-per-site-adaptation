"""Batch 14a: MOTChallenge training sequences (MOT17, MOT20; CC BY-NC-SA 3.0) as
per-site benchmarks in the paper's protocol. Each sequence is one site. Person boxes: gt.txt classes 1 (pedestrian),
2 (person on vehicle) and 7 (static person), at any visibility; other classes are background. Temporal split of the frames:
pool [0, 50%), gap [50, 60%), val [60, 70%), test [70, 100%), every fifth frame in each window (the videos run at
14 to 30 fps). Camera: static or moving, from the MOTChallenge sequence descriptions. Writes
data/mot/s<id>/{pool,gap,val,test}/{images,labels} (images are symlinks) and data/mot/manifest.json."""
import os, glob, json, collections, configparser, sys
import numpy as np
J = "/path/to/domain/journal"; RAW = f"{J}/external/mot"; OUT = f"{J}/data/mot"
SEQ = [(3001, "MOT17", "MOT17-02", "static"), (3002, "MOT17", "MOT17-04", "static"), (3003, "MOT17", "MOT17-05", "moving"),
       (3004, "MOT17", "MOT17-09", "static"), (3005, "MOT17", "MOT17-10", "moving"), (3006, "MOT17", "MOT17-11", "moving"),
       (3007, "MOT17", "MOT17-13", "moving"), (3011, "MOT20", "MOT20-01", "static"), (3012, "MOT20", "MOT20-02", "static"),
       (3013, "MOT20", "MOT20-03", "static"), (3014, "MOT20", "MOT20-05", "static")]
man = {}
for sid, ds, seq, cam in SEQ:
    d = glob.glob(f"{RAW}/{ds}/train/{seq}*")
    d = sorted(d)[0] if ds == "MOT20" else [x for x in d if x.endswith("-FRCNN")][0]
    ini = configparser.ConfigParser(); ini.read(f"{d}/seqinfo.ini"); W = int(ini["Sequence"]["imWidth"]); H = int(ini["Sequence"]["imHeight"])
    n = int(ini["Sequence"]["seqLength"]); fps = int(ini["Sequence"]["frameRate"]); ext = ini["Sequence"]["imExt"]
    A = collections.defaultdict(list)
    for l in open(f"{d}/gt/gt.txt"):
        q = l.strip().split(",")
        if int(q[7]) in (1, 2, 7): A[int(q[0])].append([float(q[2]), float(q[3]), float(q[4]), float(q[5])])
    b = [int(round(x * n)) for x in (0.5, 0.6, 0.7)]
    parts = {"pool": range(1, b[0] + 1, 5), "gap": range(b[0] + 1, b[1] + 1, 5), "val": range(b[1] + 1, b[2] + 1, 5), "test": range(b[2] + 1, n + 1, 5)}
    for p, fr in parts.items():
        o = f"{OUT}/s{sid}/{p}"; os.makedirs(f"{o}/images", exist_ok=True); os.makedirs(f"{o}/labels", exist_ok=True)
        for f in fr:
            src = f"{d}/img1/{f:06d}{ext}"; dst = f"{o}/images/{f:06d}.jpg"
            if not os.path.islink(dst): os.symlink(src, dst)
            with open(f"{o}/labels/{f:06d}.txt", "w") as fl:
                for x, y, w, h in A.get(f, []):
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                    if x2 > x1 and y2 > y1: fl.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
    tb = [b for f in parts["test"] for b in A.get(f, [])]
    man[str(sid)] = {"dataset": ds.lower(), "video": seq, "mount": cam, "size": [W, H], "fps": fps, "frames": n,
                     **{f"n_{p}": len(fr) for p, fr in parts.items()}, "test_people": len(tb),
                     "median_box": [float(np.median([b[2] for b in tb])), float(np.median([b[3] for b in tb]))] if tb else [0, 0]}
    print(sid, seq, man[str(sid)], flush=True)
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1); print("MOT_BUILD_DONE", len(man))
