#!/bin/bash
# Batch 16: Cityscapes -> Foggy Cityscapes (beta 0.02), person class. Log: logs/b16.log
cd /path/to/domain/journal/code; P=/opt/conda/bin/python; L=../logs/b16.log; export JDOMAIN=all
echo "$(date) start" >> $L
for s in unzip build source prep; do $P b16_cityscapes.py $s > ../logs/b16_$s.log 2>&1 || { echo "$(date) FAILED $s" >> $L; exit 1; }; echo "$(date) $s done: $(tail -n 1 ../logs/b16_$s.log)" >> $L; done
$P b16_cityscapes.py train 2 > ../logs/b16_train.log 2>&1 &
$P b16_cityscapes.py epochs a > ../logs/b16_epochs_a.log 2>&1 &
wait
$P b16_cityscapes.py policy > ../logs/b16_policy.log 2>&1
echo "$(date) B16_DONE $(tail -n 1 ../logs/b16_policy.log)" >> $L
