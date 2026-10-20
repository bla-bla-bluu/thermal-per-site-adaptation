#!/bin/bash
cd /path/to/domain/journal/code
P=/opt/conda/bin/python
export JDOMAIN=all
$P b6_train.py > ../logs/b6_train.log 2>&1 || { echo FAILED_B6_TRAIN; exit 1; }
for w in 0 1 2; do $P b6_epochs.py $w 3 > ../logs/b6_epochs_w$w.log 2>&1 & done; wait
$P b6_policy.py > ../results/b6_policy.txt 2>&1
echo BATCH6_DONE
