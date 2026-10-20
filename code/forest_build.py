"""Forest scenes for the journal protocol: each forest video is one target camera/scene.
Labelled frames: the authors' 'wild' labels, matched to video frames (wild_video_match.json, correlation >= 0.98);
images are the camera's own 8-bit video frames (640x512 with colour bar), labels converted from 616- to 640-px width.
Split points are the 50/60/70% quantiles of the labelled frames in time (for FLC, where every frame is labelled, this
is the same 50/10/10/30 split):
  pool     every 5th video frame (5 frames/s) before the 50% point, plus the labelled frames there (unlabeled for ST)
  pool_lab the labelled frames before the 50% point (supervised in-domain reference only)
  gap      every 5th video frame between the 50% and 60% points (unlabeled; label-free model selection)
  val      labelled frames between the 60% and 70% points
  test     labelled frames after the 70% point
Scene ids: 101-105 = fixed_1 (clip 1), fixed_2 (clip 2), fixed_3 (clip 3, the ICVGIP camera), fixed_4, fixed_5 (view 4)."""
import os, json, collections
import cv2, numpy as np
from jcommon import *
V = f"{B}/data/forest_videos"; W = f"{B}/data/wild"
SCENES = {101: ("fixed_1.mp4", "forest_clip1"), 102: ("fixed_2.mp4", "forest_clip2"), 103: ("fixed_3.mp4", "forest_clip3"),
          104: ("fixed_4.mp4", "forest_view4a"), 105: ("fixed_5.mp4", "forest_view4b")}
match = json.load(open(f"{V}/wild_video_match.json"))
lab = collections.defaultdict(dict)
for k, (vid, j, c) in match.items():
    if c >= 0.98:
        lab[vid][int(j)] = int(k)
def label_lines(w):
    out = []
    for l in open(f"{W}/label/{w}.txt"):
        q = l.split()
        if len(q) == 5:
            x, y, bw, bh = map(float, q[1:]); out.append(f"0 {x * 616 / 640:.6f} {y:.6f} {bw * 616 / 640:.6f} {bh:.6f}\n")
    return "".join(out)
man = {}
for sid, (vid, name) in SCENES.items():
    cap = cv2.VideoCapture(f"{V}/{vid}"); frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    L = sorted(lab[vid].items()); m = len(L); tj = [j for j, _ in L]
    q50, q60, q70 = tj[int(0.5 * m)], tj[int(0.6 * m)], tj[int(0.7 * m)]
    parts = {"pool": sorted(set(range(0, q50, 5)) | {j for j in tj if j < q50}), "pool_lab": [j for j in tj if j < q50],
             "gap": list(range(q50, q60, 5)), "val": [j for j in tj if q60 <= j < q70], "test": [j for j in tj if j >= q70]}
    man[sid] = {"name": name, "video": vid, "video_frames": len(frames), "labelled": m, "split_frames": [q50, q60, q70]}
    for part, js in parts.items():
        d = sdir(sid, part); os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True)
        npl = 0
        for j in js:
            cv2.imwrite(f"{d}/images/{j:05d}.jpg", frames[j], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if j in lab[vid] and part in ("pool_lab", "val", "test", "pool"):
                t = label_lines(lab[vid][j]); npl += t.count("\n")
                open(f"{d}/labels/{j:05d}.txt", "w").write(t)
        open(f"{J}/data/flc/s{sid}_{part}.yaml", "w").write(f"path: {d}\ntrain: images\nval: images\nnames:\n  0: person\n")
        man[sid][part] = {"frames": len(js), "people": npl}
    print(sid, name, {p: man[sid][p] for p in parts}, flush=True)
json.dump(man, open(f"{J}/data/forest/manifest.json", "w"), indent=1)
print("FOREST_BUILT")
