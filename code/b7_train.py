"""Batch 7b: (1) our self-training with YOLO26n (COCO-pretrained student on the YOLO26n teacher's tau=0.55 labels),
(2) RT-SFOD with its official settings (AdaBN warm start, then 60 epochs of the RT-SFOD teacher-student stage) at 640 px,
both from the same YOLO26n source model, three seeds, every epoch saved. At most 3 jobs at once."""
import os, time, subprocess, glob
from jcommon import *
PY = "/opt/conda/bin/python"; SRC26 = f"{J}/runs/J_Base_y26n/weights/last.pt"; RS = f"{J}/baselines/RT-SFOD"
env = subprocess.run(["/opt/conda/bin/conda", "env", "list"], capture_output=True, text=True).stdout
PYR = [l.split()[-1] for l in env.splitlines() if l.startswith("rtsfod")][0] + "/bin/python"
def tag(sid): return f"s{sid:02d}" if sid < 100 else f"s{sid}"
running = []
def wait(n):
    while True:
        running[:] = [(a, p) for a, p in running if p.poll() is None]
        if len(running) < n: return
        time.sleep(15)
def st26(sid, seed):
    name = f"ST26_{tag(sid)}_t055_seed{seed}"
    if os.path.exists(f"{J}/runs/{name}/DONE"): return
    code = (f"from ultralytics import YOLO; YOLO('{B}/yolo26n.pt').train(data='{J}/data/flc/{tag(sid)}_y26t055.yaml', epochs=50, imgsz=640, "
            f"batch=16, device='cuda:0', workers=4, cache='ram', project='{J}/runs', name='{name}', exist_ok=True, patience=50, seed={seed}, "
            f"plots=False, save_period=1); open('{J}/runs/{name}/DONE','w').write('ok')")
    running.append((name, subprocess.Popen([PY, "-c", code], stdout=open(f"{J}/logs/train_{name}.log", "w"), stderr=subprocess.STDOUT, cwd=J)))
def rtsfod(sid, seed):
    out = f"{J}/runs_rtsfod/{tag(sid)}_seed{seed}"
    if os.path.exists(f"{out}/DONE"): return
    yml = f"{J}/data/flc/{tag(sid)}_rtsfod.yaml"; s1 = f"{J}/runs_rtsfod/{tag(sid)}_stage1"
    cmd = (f"cd {RS} && ( [ -f {s1}/DONE ] || ( {PYR} scripts/YOLO26/stage0_stage1_adabn_rc_yolo26.py --weights {SRC26} --data {yml} "
           f"--out_dir {s1} --imgsz 640 --batch 16 --workers 4 --epochs_adabn 2 --epochs_rc 0 --device 0 && touch {s1}/DONE ) ) && "
           f"{PYR} scripts/YOLO26/stage2_rtsfod_yolo26.py --stage1_model {s1}/yolo26_stage1_adabnrc_{tag(sid)}_rtsfod.pt --data {yml} "
           f"--out_dir {out} --imgsz 640 --batch 16 --workers 4 --epochs 60 --seed {seed} --device 0 && touch {out}/DONE")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    running.append((out, subprocess.Popen(["bash", "-c", cmd], stdout=open(f"{J}/logs/rtsfod_{tag(sid)}_seed{seed}.log", "w"), stderr=subprocess.STDOUT)))
sc = [s for s, *_ in scenarios()]
for seed in (0, 1, 2):
    for sid in sc:
        wait(3); st26(sid, seed)
for sid in sc:                     # stage 1 is shared by the seeds of a scene, so run seed 0 of every scene first
    wait(3); rtsfod(sid, 0)
wait(1)
for seed in (1, 2):
    for sid in sc:
        wait(3); rtsfod(sid, seed)
wait(1)
print("B7_TRAIN_DONE")
