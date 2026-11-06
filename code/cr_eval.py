"""One matcher and one AP routine for every number.

Detections come either from the Ultralytics validator (gen_std: its dataloader, letterbox, inference and NMS at
conf >= 0.001, NMS IoU 0.5) or from sliced inference (gen_sahi: SAHI tiles plus the full-frame pass, merged by NMS at
IoU 0.5, conf >= 0.001). Both return one record per frame: {"im", "gt" (N,4), "det" (M,4), "conf" (M,), "gain"} with
boxes as x1, y1, x2, y2 in pixels. match() and summarize() score any such list the same way: detections sorted by
confidence, each matched to the best unmatched ground-truth box of its frame at IoU >= 0.5, 101-point AP. P, R and F1
at the operating point conf = 0.15 are read off the same ranked list as AP. bootstrap() resamples frames."""
import os, glob, tempfile
import numpy as np


def gen_std(weights, yaml, iou=0.5):
    """Run the Ultralytics validator on the val split of a data file; return (records, native mAP50)."""
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    rec = []
    orig = DetectionValidator._process_batch

    def patched(self, preds, batch):
        g = batch["bboxes"].detach().cpu().numpy().reshape(-1, 4) if batch["cls"].shape[0] else np.zeros((0, 4))
        p = preds["bboxes"].detach().cpu().numpy().reshape(-1, 4)
        c = preds["conf"].detach().cpu().numpy().reshape(-1)
        gain = float(batch["ratio_pad"][0][0]) if batch.get("ratio_pad") is not None else 1.0   # letterbox scale
        rec.append({"im": os.path.basename(batch["im_file"]), "gt": g, "det": p, "conf": c, "gain": gain})
        return orig(self, preds, batch)

    DetectionValidator._process_batch = patched
    try:
        r = YOLO(weights).val(data=yaml, split="val", imgsz=640, batch=16, conf=0.001, iou=iou,
                              plots=False, verbose=False, workers=4, device="cuda:0",
                              project=os.path.join(tempfile.gettempdir(), "val_runs"), name="v", exist_ok=True)
    finally:
        DetectionValidator._process_batch = orig
    return rec, float(r.box.map50)


def gen_sahi(weights, slice_=320, overlap=0.2, img_dir=None):
    """Sliced inference on every image of img_dir; labels are read from the sibling labels folder (YOLO txt)."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from PIL import Image
    lbl_dir = img_dir.replace("/images", "/labels")
    dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=weights,
                                            confidence_threshold=0.001, device="cuda:0")
    rec = []
    for p in sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png")):
        W, H = Image.open(p).size
        g = []
        lf = f"{lbl_dir}/{os.path.splitext(os.path.basename(p))[0]}.txt"
        if os.path.exists(lf):
            for line in open(lf):
                q = line.split()
                if len(q) == 5:
                    cx, cy, w, h = map(float, q[1:])
                    g.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        r = get_sliced_prediction(p, dm, slice_height=slice_, slice_width=slice_,
                                  overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                  postprocess_type="NMS", postprocess_match_metric="IOU",
                                  postprocess_match_threshold=0.5, verbose=0)
        d = [[o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy] for o in r.object_prediction_list]
        c = [o.score.value for o in r.object_prediction_list]
        rec.append({"im": os.path.basename(p), "gt": np.array(g).reshape(-1, 4),
                    "det": np.array(d).reshape(-1, 4), "conf": np.array(c), "gain": 1.0})
    return rec


def iou_mat(a, b):
    """IoU of every box of a (N,4) with every box of b (M,4); result (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def match(rec, thr=0.5):
    """Per frame: confidence-ordered greedy matching to the best unmatched GT (IoU >= thr).
    Returns per-detection (conf, tp, frame) and per-GT (frame, width_px_orig, matched_conf)."""
    dets, gts = [], []
    for fi, r in enumerate(rec):
        o = np.argsort(-r["conf"], kind="stable"); D = r["det"][o]; C = r["conf"][o]
        M = iou_mat(D, r["gt"]); used = np.zeros(len(r["gt"]), bool); mc = np.full(len(r["gt"]), -1.0)
        for k in range(len(D)):
            tp = 0
            if len(r["gt"]):
                row = np.where(used, -1, M[k]); j = int(np.argmax(row))
                if row[j] >= thr:
                    used[j] = True; mc[j] = C[k]; tp = 1
            dets.append((C[k], tp, fi))
        w = (r["gt"][:, 2] - r["gt"][:, 0]) / r["gain"] if len(r["gt"]) else np.zeros(0)
        for j in range(len(r["gt"])):
            gts.append((fi, w[j], mc[j]))
    return np.array(dets).reshape(-1, 3), np.array(gts).reshape(-1, 3)


def ap101(rec_, prec_):
    """101-point interpolated AP from recall and precision arrays (np.trapezoid needs NumPy 2)."""
    if len(rec_) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec_, [rec_[-1]], [1.0])); mpre = np.concatenate(([1.0], prec_, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def summarize(dets, n_gt, op=0.15):
    """AP50 over the ranked list and P, R, F1, TP, FP at confidence op, read off the same list."""
    o = np.argsort(-dets[:, 0], kind="stable"); s = dets[o, 0]; t = dets[o, 1]
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t)
    rec_ = ctp / max(n_gt, 1); prec_ = ctp / np.maximum(ctp + cfp, 1e-9)
    k = int((s >= op).sum())
    P = prec_[k - 1] if k else 0.0; R = rec_[k - 1] if k else 0.0
    F = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return {"AP50": ap101(rec_, prec_), "P": P, "R": R, "F1": F, "TP": int(ctp[k - 1]) if k else 0,
            "FP": int(cfp[k - 1]) if k else 0, "nGT": int(n_gt), "maxR": float(rec_[-1]) if len(rec_) else 0.0}


def bootstrap(dets, gts, n_frames, n=5000, seed=0, op=0.15):
    """95% intervals (2.5 and 97.5 percentiles) of AP50, R, P and F1 over n resamples of the frames."""
    rng = np.random.default_rng(seed)
    by_f = [dets[dets[:, 2] == f] for f in range(n_frames)]
    ng_f = np.bincount(gts[:, 0].astype(int), minlength=n_frames) if len(gts) else np.zeros(n_frames, int)
    out = {"AP50": [], "R": [], "P": [], "F1": []}
    for _ in range(n):
        idx = rng.integers(0, n_frames, n_frames)
        d = np.concatenate([by_f[i] for i in idx]) if n_frames else np.zeros((0, 3))
        m = summarize(d, ng_f[idx].sum(), op)
        for k in out:
            out[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}
