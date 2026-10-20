#!/bin/bash
# SF-UT (b10_sfut.py), called by run_batch10.sh once the SF-YOLO runs are done.
cd /path/to/domain/journal/code
PU=/path/to/envs/sfut/bin/python; P=/opt/conda/bin/python; export JDOMAIN=all
# source training may already be running (started early to overlap with SF-YOLO): wait for it
while pgrep -f "b10_sfut.py source" > /dev/null; do sleep 60; done
$PU b10_sfut.py source > ../logs/b10_sfut_source.log 2>&1
$PU b10_sfut.py adapt 2 > ../logs/b10_sfut_adapt.log 2>&1
$PU b10_sfut.py cache > ../logs/b10_sfut_cache.log 2>&1
$P b10_sfut.py policy > ../logs/b10_sfut_policy.log 2>&1
echo "$(date) run_sfut finished"
