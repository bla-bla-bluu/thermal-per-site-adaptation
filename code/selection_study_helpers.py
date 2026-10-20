"""Helpers shared by the selection and val-size studies."""
import numpy as np
from PIL import Image

def load_gt(img_paths):
    out = []
    for p in img_paths:
        W, H = Image.open(p).size; g = []
        for l in open(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"):
            q = l.split()
            if len(q) == 5:
                cx, cy, w, h = map(float, q[1:]); g.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
        out.append(np.array(g).reshape(-1, 4))
    return out

def predict(model, arrays):
    dets = []
    for i in range(0, len(arrays), 32):
        for r in model.predict(arrays[i:i+32], imgsz=640, conf=0.001, iou=0.5, max_det=300, verbose=False, device="cuda:0"):
            dets.append((r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy().reshape(-1)))
    return dets

