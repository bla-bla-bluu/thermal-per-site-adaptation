"""The woodland of forest view 4 is ONE recording whose camera was moved between two nearby positions; the camera's
export split it into two videos (fixed_4.mp4, then fixed_5.mp4). Build it as one scene, sid 106, on the concatenated
timeline, with the same split rule as forest_build.py (50/60/70% quantiles of the labelled frames in time).
The previous per-video scenes 104 and 105 are removed from the active manifest (a copy is kept as manifest_5videos.json).
FLC-Rural s04 is forest clip 3 (sid 103), so the benchmark is 20 FLC-Rural recordings and three forest recordings."""
import os, json, shutil, collections
import cv2
from jcommon import *
V = f"{B}/data/forest_videos"; W = f"{B}/data/wild"; SID = 106
VIDS = ["fixed_4.mp4", "fixed_5.mp4"]
match = json.load(open(f"{V}/wild_video_match.json"))
lab = collections.defaultdict(dict)
for k, (vid, j, c) in match.items():
    if c >= 0.98: lab[vid][int(j)] = int(k)
def label_lines(w):
    out = []
    for l in open(f"{W}/label/{w}.txt"):
        q = l.split()
        if len(q) == 5:
            x, y, bw, bh = map(float, q[1:]); out.append(f"0 {x * 616 / 640:.6f} {y:.6f} {bw * 616 / 640:.6f} {bh:.6f}\n")
    return "".join(out)
# concatenated timeline: global frame index g = offset(video) + local index
offs, n_all = {}, 0
for v in VIDS:
    cap = cv2.VideoCapture(f"{V}/{v}"); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); offs[v] = n_all; n_all += n
glab = {offs[v] + j: (v, w) for v in VIDS for j, w in lab[v].items()}
tj = sorted(glab); m = len(tj)
q50, q60, q70 = tj[int(0.5 * m)], tj[int(0.6 * m)], tj[int(0.7 * m)]
parts = {"pool": sorted(set(range(0, q50, 5)) | {j for j in tj if j < q50}), "pool_lab": [j for j in tj if j < q50],
         "gap": list(range(q50, q60, 5)), "val": [j for j in tj if q60 <= j < q70], "test": [j for j in tj if j >= q70]}
need = collections.defaultdict(set)
for part, js in parts.items():
    for g in js: need[g].add(part)
for part in parts:
    d = sdir(SID, part); os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
npl = collections.Counter()
for v in VIDS:          # stream each video once and write only the frames that are needed
    cap = cv2.VideoCapture(f"{V}/{v}"); j = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = offs[v] + j
        for part in need.get(g, ()):
            d = sdir(SID, part); cv2.imwrite(f"{d}/images/{g:05d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if g in glab:
                t = label_lines(glab[g][1]); npl[part] += t.count("\n"); open(f"{d}/labels/{g:05d}.txt", "w").write(t)
        j += 1
for part in parts:
    open(f"{J}/data/flc/s{SID}_{part}.yaml", "w").write(f"path: {sdir(SID, part)}\ntrain: images\nval: images\nnames:\n  0: person\n")
mf = f"{J}/data/forest/manifest.json"; bk = f"{J}/data/forest/manifest_5videos.json"
if not os.path.exists(bk): shutil.copy(mf, bk)
man = json.load(open(bk)); man.pop("104", None); man.pop("105", None)
man[str(SID)] = {"name": "forest_view4", "video": "+".join(VIDS), "video_frames": n_all, "labelled": m, "split_frames": [q50, q60, q70],
                 **{p: {"frames": len(js), "people": npl[p]} for p, js in parts.items()},
                 "view4b_starts_at": offs["fixed_5.mp4"]}
json.dump(man, open(mf, "w"), indent=1)
print(SID, {p: man[str(SID)][p] for p in parts}, "split", [q50, q60, q70], "view 4b starts at", offs["fixed_5.mp4"], flush=True)
print("VIEW4_BUILT")
