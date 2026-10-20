#!/bin/bash
# RT-SFOD without AdaBN (stage 2 from the source model), then cache epochs, then the per-scene policy table again.
cd /path/to/domain/journal/code
export JDOMAIN=all
until ! pgrep -f "b7_policy2.py" > /dev/null; do sleep 30; done
/opt/conda/bin/python b7b_noadabn.py train > ../logs/b7b_train.log 2>&1 || { echo FAILED_B7B_TRAIN; exit 1; }
/opt/conda/bin/python b7b_noadabn.py cache > ../logs/b7b_cache.log 2>&1 || { echo FAILED_B7B_CACHE; exit 1; }
/opt/conda/bin/python b7_policy2.py > ../logs/b7_policy2.log 2>&1 || { echo FAILED_B7_POLICY2; exit 1; }
echo BATCH7B_DONE
