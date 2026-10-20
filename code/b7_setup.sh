#!/bin/bash
# Batch 7 setup: (1) YOLO26n source model on LLVIP (same 12,025 images as the YOLOv8n baseline; 50 epochs,
# final epoch), needed by RT-SFOD, which is built on YOLO26; (2) RT-SFOD in its own conda env (AGPL-3.0 code).
J=/path/to/domain/journal; B=/path/to/domain
mkdir -p $J/baselines $J/logs
cd $J
if [ ! -f runs/J_Base_y26n/DONE ]; then
  /opt/conda/bin/python -c "from ultralytics import YOLO; YOLO('$B/yolo26n.pt').train(data='$B/code/camready/cr_llvip.yaml', epochs=50, imgsz=640, batch=64, device='cuda:0', workers=8, cache='ram', project='$J/runs', name='J_Base_y26n', exist_ok=True, patience=50, seed=0, plots=False); open('$J/runs/J_Base_y26n/DONE','w').write('ok')" > $J/logs/train_J_Base_y26n.log 2>&1 &
fi
cd $J/baselines
[ -d RT-SFOD ] || git clone -q https://github.com/Sairam13001/RT-SFOD.git
cd RT-SFOD && git log --oneline -1
if ! /opt/conda/bin/conda env list | grep -q rtsfod; then
  /opt/conda/bin/conda create -y -q -n rtsfod python=3.10 > $J/logs/rtsfod_env.log 2>&1
fi
P=$(/opt/conda/bin/conda env list | awk '/rtsfod/ {print $NF}')/bin/python
$P -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124 >> $J/logs/rtsfod_env.log 2>&1
$P -m pip install -q -e . >> $J/logs/rtsfod_env.log 2>&1
$P -c "import torch, ultralytics; print('rtsfod env ok', torch.__version__, torch.cuda.is_available(), ultralytics.__version__, ultralytics.__file__)"
ls scripts scripts/YOLO26 2>/dev/null | head -20
wait
echo B7_SETUP_DONE
