#!/bin/bash
# Batch 14: MOTChallenge sequences, COCO source. Waits for the downloads and for batch 12 to finish. Log: logs/b14.log
cd /path/to/domain/journal/code; P=/opt/conda/bin/python; L=../logs/b14.log; export JDOMAIN=all; M=../external/mot
until [ -f $M/dl.done ]; do sleep 120; done
cd $M && for z in MOT17 MOT20; do [ -d $z ] || unzip -q $z.zip; done; cd /path/to/domain/journal/code
echo "$(date) unzipped: $(ls $M)" >> $L
$P b14_mot_build.py > ../logs/b14_build.log 2>&1 && echo "$(date) built" >> $L
until grep -q B12_DONE ../logs/b12.log 2>/dev/null; do sleep 300; done
$P b14_mot.py prep > ../logs/b14_prep.log 2>&1 && echo "$(date) prep done" >> $L
$P b14_mot.py train 3 > ../logs/b14_train.log 2>&1 &
for w in a b; do $P b14_mot.py epochs $w > ../logs/b14_epochs_$w.log 2>&1 & done
wait
$P b14_mot.py policy > ../logs/b14_policy.log 2>&1
echo "$(date) B14_DONE" >> $L
