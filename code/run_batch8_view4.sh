#!/bin/bash
# Batch 8: rebuild the woodland of forest view 4 as ONE recording (sid 106, camera moved between two positions) and run
# every experiment of the paper on it; then recompute the all-scene analyses over the 23 recordings.
cd /path/to/domain/journal/code
P=/opt/conda/bin/python; PYR=/path/to/envs/rtsfod/bin/python
$P forest_build_view4.py > ../logs/b8_build.log 2>&1 || { echo FAILED_BUILD; exit 1; }
export JDOMAIN=forest JSIDS=106
$P teacher_and_baseline.py > ../logs/b8_teacher.log 2>&1 || { echo FAILED_TEACHER; exit 1; }
$P train_flc.py > ../logs/b8_train.log 2>&1 || { echo FAILED_TRAIN; exit 1; }
$P eval_flc.py > ../logs/b8_eval.log 2>&1 || { echo FAILED_EVAL; exit 1; }
$P batch3_pseudolabels.py > ../logs/b8_variants.log 2>&1 || { echo FAILED_VARIANTS; exit 1; }
$P selection_study.py 0 1 > ../logs/b8_sel.log 2>&1 || { echo FAILED_SEL; exit 1; }
$P valsize_study.py > ../logs/b8_valsize.log 2>&1 || { echo FAILED_VALSIZE; exit 1; }
$P b6_train.py > ../logs/b8_b6train.log 2>&1 || { echo FAILED_B6TRAIN; exit 1; }
$P b6_epochs.py 0 1 > ../logs/b8_b6epochs.log 2>&1 || { echo FAILED_B6EPOCHS; exit 1; }
$P b7_prep.py > ../logs/b8_b7prep.log 2>&1 || { echo FAILED_B7PREP; exit 1; }
$P b7_train.py > ../logs/b8_b7train.log 2>&1 || { echo FAILED_B7TRAIN; exit 1; }
$P b7_epochs.py st26 0 1 > ../logs/b8_b7ep_st26.log 2>&1 || { echo FAILED_B7EP_ST26; exit 1; }
$PYR b7_epochs.py rtsfod 0 1 > ../logs/b8_b7ep_rt.log 2>&1 || { echo FAILED_B7EP_RT; exit 1; }
$P b7b_noadabn.py train > ../logs/b8_b7b_train.log 2>&1 || { echo FAILED_B7B_TRAIN; exit 1; }
$P b7b_noadabn.py cache > ../logs/b8_b7b_cache.log 2>&1 || { echo FAILED_B7B_CACHE; exit 1; }
unset JSIDS; export JDOMAIN=all
$P j_unify_eval.py > ../logs/b8_unify.log 2>&1 || { echo FAILED_UNIFY; exit 1; }
$P gonogo_analysis.py > ../results/gonogo_report_all.txt 2> ../logs/b8_gonogo.err || { echo FAILED_GONOGO; exit 1; }
$P b6_policy2.py > ../results/b6_policy2.txt 2> ../logs/b8_b6policy2.err || { echo FAILED_B6POLICY2; exit 1; }
$P b7_policy2.py > ../logs/b7_policy2.log 2>&1 || { echo FAILED_B7POLICY2; exit 1; }
echo BATCH8_DONE
