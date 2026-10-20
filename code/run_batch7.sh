#!/bin/bash
cd /path/to/domain/journal/code
P=/opt/conda/bin/python; export JDOMAIN=all
until [ -f ../runs/J_Base_y26n/DONE ]; do sleep 60; done
PYR=$(/opt/conda/bin/conda env list | awk '/^rtsfod/ {print $NF}')/bin/python
$P b7_prep.py > ../logs/b7_prep.log 2>&1 || { echo FAILED_B7_PREP; exit 1; }
$P b7_train.py > ../logs/b7_train.log 2>&1 || { echo FAILED_B7_TRAIN; exit 1; }
for w in 0 1 2; do $P b7_epochs.py st26 $w 3 > ../logs/b7_epochs_st26_w$w.log 2>&1 & done
for w in 0 1; do $PYR b7_epochs.py rtsfod $w 2 > ../logs/b7_epochs_rtsfod_w$w.log 2>&1 & done
wait
$P b7_policy.py > ../results/b7_policy.txt 2>&1
echo BATCH7_DONE
