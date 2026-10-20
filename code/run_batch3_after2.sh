#!/bin/bash
# Start batch 3 as soon as batch 2 has finished, so the GPU never idles.
cd /path/to/domain/journal/code
until grep -q BATCH2_DONE ../logs/batch2.log 2>/dev/null; do sleep 60; done
/opt/conda/bin/python analyze_selection.py > ../results/selection_summary.txt 2>&1
/opt/conda/bin/python batch3_pseudolabels.py > ../logs/b3.log 2>&1 && echo BATCH3_DONE || echo FAILED_BATCH3
