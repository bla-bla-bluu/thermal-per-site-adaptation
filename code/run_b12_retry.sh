#!/bin/bash
# Batch 12 retry: some students of the first pass died (CUDA out of memory, shown as an NVML assertion; and runs that
# read yolov8s.pt while it was still being downloaded). After the first pass ends, delete the failed run folders and train
# them again, two at a time. The caching worker of run_b12b.sh picks them up.
cd /path/to/domain/journal/code; P=/opt/conda/bin/python; L=../logs/b12.log; export JDOMAIN=all
while kill -0 3297393 2>/dev/null; do sleep 120; done
for d in ../runs/B12*; do [ -f $d/DONE ] || { echo "$(date) retry $(basename $d)" >> $L; rm -rf $d; }; done
$P b12_sources.py train 2 > ../logs/b12_train_retry.log 2>&1
echo "$(date) retry pass finished: $(ls -d ../runs/B12*/DONE | wc -l) runs done" >> $L
