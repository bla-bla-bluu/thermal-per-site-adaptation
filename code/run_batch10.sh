#!/bin/bash
# Batch 10: external benchmarks (BIRDSAI, CAMEL), SF-YOLO and SF-UT.
# Waits until the GPU is free of other work (the vehicle-paper queue) then runs everything.
cd /path/to/domain/journal/code
P=/opt/conda/bin/python; PS=/path/to/envs/sfyolo/bin/python; export JDOMAIN=all
L=../logs/batch10.log; echo "$(date) waiting for a free GPU" >> $L
free=0
while [ $free -lt 10 ]; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  sched=$(ps -eo cmd | grep -c "[s]cripts/scripts/scheduler.py")
  if [ "$apps" -eq 0 ] && [ "$sched" -eq 0 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done
echo "$(date) GPU free, starting" >> $L
# 1. external benchmarks: teacher, students, per-epoch caches, policy
$P b10_ext_pipeline.py prep > ../logs/b10_ext_prep.log 2>&1 || { echo "$(date) FAILED ext prep" >> $L; exit 1; }
$P b10_ext_pipeline.py train 4 > ../logs/b10_ext_train.log 2>&1 &
for w in a b; do $P b10_ext_pipeline.py epochs $w > ../logs/b10_ext_epochs_$w.log 2>&1 & done
wait
$P b10_ext_pipeline.py policy > ../logs/b10_ext_policy.log 2>&1
echo "$(date) EXT_DONE" >> $L
# 2. SF-YOLO
$PS b10_sfyolo.py source > ../logs/b10_sfyolo_source.log 2>&1
$PS b10_sfyolo.py tam > ../logs/b10_sfyolo_tam.log 2>&1
$PS b10_sfyolo.py adapt 3 > ../logs/b10_sfyolo_adapt.log 2>&1
for w in 0 1; do $PS b10_sfyolo.py cache $w 2 > ../logs/b10_sfyolo_cache_$w.log 2>&1 & done
wait
$P b10_sfyolo.py policy > ../logs/b10_sfyolo_policy.log 2>&1
echo "$(date) SFYOLO_DONE" >> $L
# 3. SF-UT (runs only if its environment and script are ready)
if [ -f ../baselines/SFUT_READY ]; then
  bash run_sfut.sh >> $L 2>&1
  echo "$(date) SFUT_DONE" >> $L
fi
echo "$(date) BATCH10_DONE" >> $L
