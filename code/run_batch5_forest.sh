#!/bin/bash
# Journal batch 5: the five forest videos through the same per-scene protocol as FLC-Rural, then the
# go/no-go analysis over all 24 scenes.
cd /path/to/domain/journal/code
P=/opt/conda/bin/python
$P forest_build.py > ../logs/b5_build.log 2>&1 || { echo FAILED_BUILD; exit 1; }
export JDOMAIN=forest
$P teacher_and_baseline.py > ../logs/b5_teacher.log 2>&1 || { echo FAILED_TEACHER; exit 1; }
$P train_flc.py > ../logs/b5_train.log 2>&1 || { echo FAILED_TRAIN; exit 1; }
$P eval_flc.py > ../logs/b5_eval.log 2>&1 || { echo FAILED_EVAL; exit 1; }
$P batch3_pseudolabels.py > ../logs/b5_variants.log 2>&1 || { echo FAILED_VARIANTS; exit 1; }
$P selection_study.py 0 2 > ../logs/b5_sel_w0.log 2>&1 &
$P selection_study.py 1 2 > ../logs/b5_sel_w1.log 2>&1 &
wait
$P valsize_study.py > ../results/valsize_report_forest.txt 2> ../logs/b5_valsize.err
JDOMAIN=all $P gonogo_analysis.py > ../results/gonogo_report_all.txt 2>&1
echo BATCH5_DONE
