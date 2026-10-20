#!/bin/bash
# Journal batch 1: FLC-Rural multi-scenario benchmark (splits -> teacher/baseline -> training -> evaluation).
cd /path/to/domain/journal/code
P=/opt/conda/bin/python
$P build_flc_splits.py   > ../logs/b1_splits.log  2>&1 || { echo FAILED_SPLITS; exit 1; }
$P teacher_and_baseline.py > ../logs/b1_teacher.log 2>&1 || { echo FAILED_TEACHER; exit 1; }
$P train_flc.py          > ../logs/b1_train.log   2>&1 || { echo FAILED_TRAIN; exit 1; }
$P eval_flc.py           > ../logs/b1_eval.log    2>&1 || { echo FAILED_EVAL; exit 1; }
echo BATCH1_DONE
