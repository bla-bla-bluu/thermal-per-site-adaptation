#!/bin/bash
# After batch 9: wait for training (BATCH9_DONE) and the three caching workers (b9_epochs2.py), then the threshold-sweep
# evaluation, SED and cross-session test (b9_after), all policy analyses (b9_policy) and the cost figures (b9_cost).
cd /path/to/domain/journal/code
P=/opt/conda/bin/python; export JDOMAIN=all
until grep -q BATCH9_DONE ../logs/batch9.log 2>/dev/null && [ -f ../logs/B9_EPOCHS2_DONE_a ] && [ -f ../logs/B9_EPOCHS2_DONE_b ] && [ -f ../logs/B9_EPOCHS2_DONE_c ]; do sleep 120; done
$P b9_after.py > ../logs/b9_after.log 2>&1 || { echo FAILED_B9_AFTER > ../logs/batch9_after.log; exit 1; }
NPROC=16 $P b9_policy.py > ../logs/b9_policy.log 2>&1 || { echo FAILED_B9_POLICY > ../logs/batch9_after.log; exit 1; }
$P b9_cost.py > ../logs/b9_cost.log 2>&1
echo BATCH9_ALL_DONE > ../logs/batch9_after.log
