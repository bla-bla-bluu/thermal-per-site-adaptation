#!/bin/bash
# Batch 12: other source detectors (YOLOv8-Small on LLVIP; COCO YOLOv8-Nano, person class). Log: logs/b12.log
cd /path/to/domain/journal/code; P=/opt/conda/bin/python; L=../logs/b12.log; export JDOMAIN=all
echo "$(date) start" >> $L
$P b12_sources.py prep v8s > ../logs/b12_prep_v8s.log 2>&1 && echo "$(date) prep v8s done" >> $L
$P b12_sources.py prep coco > ../logs/b12_prep_coco.log 2>&1 && echo "$(date) prep coco done" >> $L
$P b12_sources.py train 3 > ../logs/b12_train.log 2>&1 &
for w in 0 1; do $P b12_sources.py epochs $w 2 > ../logs/b12_epochs_$w.log 2>&1 & done
wait
for s in v8s coco; do $P b12_sources.py policy $s > ../logs/b12_policy_$s.log 2>&1; done
echo "$(date) B12_DONE" >> $L
