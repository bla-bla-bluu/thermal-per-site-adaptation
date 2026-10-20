#!/bin/bash
cd /path/to/domain/journal/code
until grep -q -E "BATCH3_DONE|FAILED_BATCH3" ../logs/batch3.log 2>/dev/null; do sleep 120; done
/opt/conda/bin/python gonogo_analysis.py > ../results/gonogo_report.txt 2>&1; echo B4_DONE >> ../results/gonogo_report.txt
