#!/bin/bash
# Batch 17: a larger source detector. YOLOv8-Medium (25.9 M parameters) trained on LLVIP like theta0
# (50 epochs, 640 px, batch 16, Ultralytics defaults, checkpoint with the best LLVIP validation fitness), then the paper's
# pipeline with YOLOv8-Medium students from COCO (b12_sources.py, B12_SRCS=v8m), 3 seeds, every epoch cached, guard.
# One student at a time. Log: logs/b17.log
cd /path/to/domain/journal; P=/opt/conda/bin/python; L=/path/to/domain/journal/logs/b17.log; export JDOMAIN=all B12_SRCS=v8m
[ -f yolov8m.pt ] || curl -s -L -o yolov8m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
echo "$(date) start; yolov8m.pt $(ls -la yolov8m.pt | awk '{print $5}') bytes" >> $L
if [ ! -f runs/J_Base_v8m/DONE ]; then
  $P -c "from ultralytics import YOLO; YOLO('/path/to/domain/journal/yolov8m.pt').train(data='/path/to/domain/code/camready/cr_llvip.yaml', epochs=50, imgsz=640, batch=16, device='cuda:0', workers=6, project='/path/to/domain/journal/runs', name='J_Base_v8m', exist_ok=True, seed=0, plots=False); open('/path/to/domain/journal/runs/J_Base_v8m/DONE','w').write('ok')" > logs/b17_source.log 2>&1
fi
[ -f runs/J_Base_v8m/DONE ] || { echo "$(date) FAILED source" >> $L; exit 1; }
echo "$(date) source done" >> $L
cd code
$P b12_sources.py prep v8m > ../logs/b17_prep.log 2>&1 && echo "$(date) prep done" >> $L || { echo "$(date) FAILED prep" >> $L; exit 1; }
rm -f ../runs/B12_TRAIN_DONE
$P b12_sources.py train 1 > ../logs/b17_train.log 2>&1 &
$P b12_sources.py epochs 0 1 > ../logs/b17_epochs.log 2>&1 &
wait
$P b12_sources.py policy v8m > ../logs/b17_policy.log 2>&1
echo "$(date) B17_DONE $(tail -n 1 ../logs/b17_policy.log)" >> $L
