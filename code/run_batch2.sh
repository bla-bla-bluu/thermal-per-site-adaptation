#!/bin/bash
# Journal batch 2: label-free selection study (3 workers) in parallel with source-initialised students.
cd /path/to/domain/journal/code
P=/opt/conda/bin/python
for w in 0 1 2; do $P selection_study.py $w 3 > ../logs/b2_sel_w$w.log 2>&1 & done
$P train_eval_srcinit.py > ../logs/b2_srcinit.log 2>&1 &
wait
echo BATCH2_DONE
