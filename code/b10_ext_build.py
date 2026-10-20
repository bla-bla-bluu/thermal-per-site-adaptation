"""Batch 10b: external benchmarks in the per-recording protocol of the paper.
  birdsai  BIRDSAI (Bondi et al., WACV 2020; LILA BC): aerial night-time thermal videos over national parks; the real videos
           with at least 300 frames that contain people (class 1); animals count as background
  camel    CAMEL (Gebhardt and Wolf, AVSS 2018): ground-level thermal sequences (FLIR Vue Pro, 336x256, 30 fps) whose notes
           mention a tripod; class 1 = person, other classes count as background. The notes of Seq11, Seq13 and Seq15 say the
           tripod was held in the hand (moving camera): these are kept, with mount = "handheld" in the manifest, and the
           paper's main CAMEL results use the stationary ones (mount = "tripod")
Labelled frames: frames with at least one annotation row (BIRDSAI) or every frame of the annotated span (CAMEL, whose
annotation covers every frame). Temporal split as for the forest videos of the paper: boundaries at the 50, 60 and 70%
quantiles of the labelled frames' times; pool and gap take every fifth video frame of their windows; validation and test
take every fifth labelled frame of theirs. Writes data/ext/<ds>/<vid>/{pool,gap,val,test}/{images,labels} (images are
symlinks; labels YOLO txt, class 0 = person; pool labels only for labelled frames) and data/ext/manifest.json."""
import os, glob, csv, json, collections, sys
import numpy as np
from PIL import Image
J = "/path/to/domain/journal"; OUT = f"{J}/data/ext"; RAW = f"{J}/external"

def birdsai():
    R = f"{RAW}/birdsai/raw"; vids = []
    for sp in ("TrainReal", "TestReal"):
        for f in sorted(glob.glob(f"{R}/{sp}/annotations/*.csv")):
            v = os.path.basename(f)[:-4]; A = collections.defaultdict(list); lab = set()
            for r in csv.reader(open(f)):
                if not r or not r[0].strip().isdigit(): continue
                fn = int(r[0]); lab.add(fn)
                if int(r[6]) == 1: A[fn].append([float(r[2]), float(r[3]), float(r[4]), float(r[5])])
            if sum(1 for k in A if A[k]) < 300: continue
            ims = {int(os.path.splitext(p)[0].split("_")[-1]): p for p in glob.glob(f"{R}/{sp}/images/{v}/*.jpg")}
            vids.append((v, ims, A, lab))
    return vids

def camel():
    R = f"{RAW}/camel/seq"; vids = []
    for d in sorted(glob.glob(f"{R}/Seq*"), key=lambda x: int(os.path.basename(x)[3:])):
        s = os.path.basename(d); info = open(f"{d}/{s}-info.txt", errors="ignore").read() if os.path.exists(f"{d}/{s}-info.txt") else ""
        if "stationary" not in info.lower() and "tripod" not in info.lower(): continue           # fixed cameras only
        imd = f"{d}/ir"
        if not os.path.isdir(imd):
            os.makedirs(imd, exist_ok=True); os.system(f"cd {imd} && unzip -q -o ../ir.zip")
        ims = {int(os.path.splitext(os.path.basename(p))[0]): p for p in glob.glob(f"{imd}/**/*.jpg", recursive=True)}
        A = collections.defaultdict(list); rows = [r for r in csv.reader(open(f"{d}/{s}-IR.txt"), delimiter="\t") if len(r) >= 7]
        for r in rows:
            if int(float(r[2])) == 1: A[int(r[0])].append([float(r[3]), float(r[4]), float(r[5]), float(r[6])])
        if not rows or sum(1 for k in A if A[k]) < 300: continue
        fr = sorted(ims); lo, hi = min(int(r[0]) for r in rows), max(int(r[0]) for r in rows)
        lab = {f for f in fr if lo <= f <= hi}                                                     # every frame of the annotated span
        mount = "handheld" if any(w in info.lower() for w in ("in hand", "handheld", "holding tripod")) else "tripod"
        vids.append((s, ims, A, lab, mount))
    return vids

def build(ds, vids, base_id):
    man = {}
    for n, (v, ims, A, lab, *mount) in enumerate(vids):
        sid = base_id + n; lab = sorted(f for f in lab if f in ims)
        q = np.quantile(lab, [0.5, 0.6, 0.7]); b1, b2, b3 = int(q[0]), int(q[1]), int(q[2])
        allf = sorted(f for f in ims if lab[0] <= f <= lab[-1])
        parts = {"pool": [f for f in allf if f < b1][::5], "gap": [f for f in allf if b1 <= f < b2][::5],
                 "val": [f for f in lab if b2 <= f < b3][::5], "test": [f for f in lab if f >= b3][::5]}
        labset = set(lab)
        for p, frames in parts.items():
            d = f"{OUT}/{ds}/s{sid}/{p}"; os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
            for f in frames:
                src = ims[f]; dst = f"{d}/images/{f:06d}.jpg"
                if not os.path.islink(dst): os.symlink(src, dst)
                if f in labset:
                    W, H = Image.open(src).size
                    open(f"{d}/labels/{f:06d}.txt", "w").write("".join(
                        f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n" for x, y, w, h in A.get(f, [])))
        W, H = Image.open(ims[lab[0]]).size
        man[str(sid)] = {"dataset": ds, "video": v, "mount": mount[0] if mount else "aerial", "size": [W, H], **{f"n_{p}": len(fr) for p, fr in parts.items()},
                         "test_people": int(sum(len(A.get(f, [])) for f in parts["test"])),
                         "median_box": [float(np.median([b[2] for f in parts["test"] for b in A.get(f, [])] or [0])),
                                        float(np.median([b[3] for f in parts["test"] for b in A.get(f, [])] or [0]))]}
        print(ds, sid, v, man[str(sid)], flush=True)
    return man

if __name__ == "__main__":
    man = {}
    if "birdsai" in sys.argv[1:]: man.update(build("birdsai", birdsai(), 1001))
    if "camel" in sys.argv[1:]: man.update(build("camel", camel(), 2001))
    mf = f"{OUT}/manifest.json"; old = json.load(open(mf)) if os.path.exists(mf) else {}
    old.update(man); json.dump(old, open(mf, "w"), indent=1); print("EXT_BUILD_DONE", len(old))
