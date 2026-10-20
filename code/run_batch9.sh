#!/bin/bash
# Batch 9: pseudo-label sets, training queue, per-epoch caches, quick analyses.
cd /path/to/domain/journal/code
P=/opt/conda/bin/python; export JDOMAIN=all
$P b9_prep.py > ../logs/b9_prep.log 2>&1 || { echo FAILED_B9_PREP > ../logs/batch9.log; exit 1; }
$P b9_train.py 4 > ../logs/b9_train.log 2>&1 &
for w in 0 1; do $P b9_epochs.py $w 2 > ../logs/b9_epochs_w$w.log 2>&1 & done
$P b9_quick.py > ../logs/b9_quick.log 2>&1
wait
echo BATCH9_DONE > ../logs/batch9.log
