# Per-site self-training of thermal person detectors

Scripts for adapting a thermal person detector to a new site from unlabelled footage by self-training, and for choosing the deployed model with a few labelled frames.

## Layout
- `code/jcommon.py`: shared paths and split logic.
- `code/b6_*`, `b7_*`, `b9_*`: pseudo-labelling, student training, per-epoch prediction caching and policy evaluation.
- `code/b10_*`: public thermal datasets, SF-YOLO and SF-UT runs, calibration.
- `code/b11_*`: label-free variants and checkpoint-selection studies.
- `code/b12_*`, `b14_*`, `b16_*`, `run_b17.sh`: other source detectors, MOTChallenge, Cityscapes to Foggy Cityscapes.
- `code/run_*.sh`: batch drivers.

## Setup
- Python with Ultralytics 8.4, SAHI 0.12 and NumPy; detectron2 0.6 for the SF-UT scripts.
- Set the data and output root: replace `/path/to/domain` and `/path/to/envs` in the scripts, or edit `jcommon.py`.

## Data
Not included. Public sets (CAMEL, BIRDSAI, MOTChallenge, Cityscapes) come from their authors under their licences; the rural recordings are on Zenodo, https://doi.org/10.5281/zenodo.22744783.

## Licence
Not set yet.
