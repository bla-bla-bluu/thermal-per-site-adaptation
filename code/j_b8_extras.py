"""After batch 8 (23 recordings: 20 FLC-Rural incl. s04 = forest clip 3 = sid 103, and three forest recordings 101, 102,
106 = view 4 as one recording): scene statistics and the scene mosaic. Writes results/scene_stats_23.json and
results/fig_scenes_23.jpg. The labelled-frame study is j_b8_valsize.py, the false pseudo-label types j_b8_fp.py."""
import os, glob, json, pickle, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
os.environ["JDOMAIN"] = "all"
from jcommon import *
import cr_eval as E
from selection_study_helpers import load_gt
def tag(s): return f"s{s:02d}" if s < 100 else f"s{s}"
SHOW = {103: "s04", 106: "s104"}
show = lambda s: SHOW.get(s, tag(s))
SIDS = [s for s, *_ in scenarios()]; print("scenes", len(SIDS), SIDS, flush=True)
FLC = [s for s in SIDS if s < 100 or s == 103]; FOR = [s for s in SIDS if s not in FLC]
R = f"{J}/results"
# ---- scene statistics
st = {}
for s in SIDS:
    d = {p: sorted(glob.glob(f"{sdir(s, p)}/images/*.jpg")) for p in ("pool", "gap", "val", "test")}
    g = load_gt(d["test"]); wh = np.concatenate([np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) for b in g if len(b)])
    W, H = Image.open(d["test"][0]).size
    st[s] = {"pool_frames": len(d["pool"]), "gap_frames": len(d["gap"]), "val_frames": len(d["val"]), "test_frames": len(d["test"]),
             "test_people": int(sum(len(b) for b in g)), "median_w": float(np.median(wh[:, 0])), "median_h": float(np.median(wh[:, 1])), "W": W, "H": H}
json.dump(st, open(f"{R}/scene_stats_23.json", "w"), indent=1); print("stats done", flush=True)
# ---- scene mosaic (one test frame per recording: the labelled frame nearest the middle of the test window with a person)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
except Exception: font = ImageFont.load_default()
order = [1, 2, 3, 20, 103, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 101, 102, 106]
TW, TH, cols = 405, 300, 6; rows_n = (len(order) + cols - 1) // cols
mos = Image.new("RGB", (cols * TW, rows_n * TH), "white")
for i, s in enumerate(order):
    ps = sorted(glob.glob(f"{sdir(s, 'test')}/images/*.jpg")); g = load_gt(ps); mid = len(ps) // 2
    k = min((j for j in range(len(ps)) if len(g[j])), key=lambda j: abs(j - mid))
    im = Image.open(ps[k]).convert("RGB"); dr = ImageDraw.Draw(im)
    for b in g[k]: dr.rectangle([float(v) for v in b], outline=(30, 210, 60), width=3)
    w, h = im.size; sc = TH / h; im = im.resize((int(w * sc), TH))
    tile = Image.new("RGB", (TW, TH), "black"); x0 = (TW - im.width) // 2
    tile.paste(im.crop((max(-x0, 0), 0, max(-x0, 0) + min(TW, im.width), TH)), (max(x0, 0), 0))
    td = ImageDraw.Draw(tile); td.rectangle([0, 0, 92, 34], fill="black"); td.text((6, 3), show(s), fill="white", font=font)
    mos.paste(tile, ((i % cols) * TW, (i // cols) * TH))
mos.save(f"{R}/fig_scenes_23.jpg", quality=88); print("mosaic", mos.size, flush=True)
print("EXTRAS_DONE")
