"""Shared paths and the fixed split protocol for the journal experiments (decided before any result was seen).
Each FLC-Rural scenario is one target camera/scene. Frames are temporally ordered video frames (checked: consecutive
frames differ 3-10x less than random pairs). Temporal split of a scenario with N frames:
  pool [0, .5N)  unlabeled adaptation footage (its labels are used only for the supervised in-domain reference
                 and to measure pseudo-label quality)
  gap  [.5N, .6N) never trained on; unlabeled frames for label-free model selection
  val  [.6N, .7N) small labelled validation split (only for the labelled-selection comparison)
  test [.7N, N)  untouched test split
Scenario 4 is excluded here: it is forest clip 3 and enters the benchmark once, as forest scene 103 (shown as s04 in the
paper). The journal benchmark is JDOMAIN=all: 19 FLC-Rural scenes + forest 101, 102, 103 and 106 (view 4, one recording)."""
import os, sys, csv, collections
B = "/path/to/domain"; J = f"{B}/journal"; FLC = "/path/to/camo/FLC-Rural"; IRD = "/path/to/camo/ir_dataset"
sys.path.insert(0, f"{B}/code/camready")
BASELINE = f"{B}/weights/baseline_best.pt"; COCO_N = f"{B}/yolov8n.pt"
SPLITS = ("pool", "gap", "val", "test")
EXCLUDE = {4}

def scenarios():
    """Target scenes of the domain chosen by the JDOMAIN environment variable: 'flc' (default; 19 FLC-Rural scenes),
    'forest' (forest videos 101, 102, 103 and 106, from data/forest/manifest.json; forest_build.py, forest_build_view4.py)
    or 'all'. FLC entries are
    (sid, name, split, [source image paths], [source label paths]) in temporal order; forest entries carry None paths."""
    dom = os.environ.get("JDOMAIN", "flc")
    out = []
    if dom in ("flc", "all"):
        out += flc_scenarios()
    if dom in ("forest", "all"):
        import json
        man = json.load(open(f"{J}/data/forest/manifest.json"))
        out += [(int(k), v["name"], "forest", None, None) for k, v in sorted(man.items(), key=lambda kv: int(kv[0]))]
    ids = os.environ.get("JSIDS")          # optional: restrict to some scene ids, e.g. JSIDS=106
    if ids:
        keep = {int(x) for x in ids.split(",")}; out = [o for o in out if o[0] in keep]
    return out


def flc_scenarios():
    """[(sid, name, split, [source image paths], [source label paths])] in temporal order."""
    rows = list(csv.DictReader(open(f"{FLC}/metadata/scenario_manifest.csv")))
    groups = collections.OrderedDict(); ctr = {"train": 0, "test": 0}
    for r in rows:
        ctr[r["split"]] += 1
        groups.setdefault((int(r["scenario_id"]), r["scenario_name"], r["split"]), []).append(ctr[r["split"]])
    out = []
    for (sid, name, split), idx in groups.items():
        if sid in EXCLUDE:
            continue
        if split == "train":
            imgs = [f"{B}/yolo_ir_oracle/train/images/{k:06d}.jpg" for k in idx]
        else:
            sub = [x for x in os.listdir(f"{B}/yolo_flc") if x.startswith(f"s{sid}_")][0]
            imgs = [f"{B}/yolo_flc/{sub}/images/{k:06d}.jpg" for k in idx]
        labs = [f"{IRD}/{split}/labels/{k:06d}.txt" for k in idx]
        out.append((sid, name, split, imgs, labs))
    return out

def split_bounds(n):
    a, b, c = int(0.5 * n), int(0.6 * n), int(0.7 * n)
    return {"pool": (0, a), "gap": (a, b), "val": (b, c), "test": (c, n)}

def sdir(sid, part):
    """Split folder of a scene; forest scenes have sid >= 100. All data yamls live in data/flc/ (s<sid>_<kind>.yaml)."""
    return f"{J}/data/forest/s{sid}/{part}" if sid >= 100 else f"{J}/data/flc/s{sid:02d}/{part}"
