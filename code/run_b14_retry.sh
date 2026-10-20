#!/bin/bash
# Batch 14 retry: 9 runs (EXne s3002 and s3013, EXsup s3014) stopped on a network error (ConnectionResetError while
# Ultralytics fetched an asset). Delete them, train again, cache their epochs, recompute the policies.
cd /path/to/domain/journal/code; P=/opt/conda/bin/python; L=../logs/b14.log; export JDOMAIN=all
for d in ../runs_mot/EX*; do [ -f $d/DONE ] || { echo "$(date) retry $(basename $d)" >> $L; rm -rf $d; }; done
rm -f ../runs_mot/EXT_TRAIN_DONE
$P b14_mot.py train 3 > ../logs/b14_train_retry.log 2>&1 &
$P b14_mot.py epochs r > ../logs/b14_epochs_r.log 2>&1 &
wait
$P b14_mot.py policy > ../logs/b14_policy.log 2>&1
echo "$(date) B14_RETRY_DONE" >> $L
