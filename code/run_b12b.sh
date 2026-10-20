#!/bin/bash
# Batch 12, continuation: per-epoch caching with one worker, then the policies once training and caching are complete.
# Writes B12_DONE to logs/b12.log as before.
cd /path/to/domain/journal/code; P=/opt/conda/bin/python; L=../logs/b12.log; export JDOMAIN=all
echo "$(date) caching restarted (run_b12b.sh)" >> $L
$P b12_sources.py epochs 0 1 > ../logs/b12_epochs_r0.log 2>&1
for s in v8s coco; do $P b12_sources.py policy $s > ../logs/b12_policy_$s.log 2>&1; done
echo "$(date) B12_DONE" >> $L
