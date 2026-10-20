#!/bin/bash
# Batch 17, second part: runs the students, caching, and policy once the source model and the preparation of run_b17.sh are done.
cd /path/to/domain/journal/code; P=/opt/conda/bin/python; L=/path/to/domain/journal/logs/b17.log; export JDOMAIN=all B12_SRCS=v8m
echo "$(date) prep done (log path fixed, continuing with the students)" >> $L
rm -f ../runs/B12_TRAIN_DONE
$P b12_sources.py train 1 > ../logs/b17_train.log 2>&1 &
$P b12_sources.py epochs 0 1 > ../logs/b17_epochs.log 2>&1 &
wait
$P b12_sources.py policy v8m > ../logs/b17_policy.log 2>&1
echo "$(date) B17_DONE $(tail -n 1 ../logs/b17_policy.log)" >> $L
