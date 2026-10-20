# Per-site self-training of thermal person detectors

Scripts for adapting a thermal person detector to a new site from unlabelled footage by self-training, and for choosing the deployed model with a few labelled frames.

## Layout
- `code/jcommon.py`: shared paths, scene list and the temporal split (pool 50%, gap 10%, val 10%, test 30%).
- `code/build_flc_splits.py`, `forest_build.py`, `forest_build_view4.py`: per-recording split folders and YOLO data files.
- `code/teacher_and_baseline.py`, `batch3_pseudolabels.py`, `train_flc.py`, `eval_flc.py`: sliced-inference teacher, pseudo-labels, students, evaluation.
- `code/b6_*`, `b7_*`, `b9_*`: students with every epoch saved, per-epoch prediction caches and policy evaluation.
- `code/b10_*`: public thermal datasets, SF-YOLO and SF-UT, calibration.
- `code/b11_*`: label-free variants and checkpoint selection.
- `code/b12_*`, `b14_*`, `b16_*`: other source detectors, MOTChallenge, Cityscapes to Foggy Cityscapes.
- `code/b7_setup.sh`: YOLO26n source model and the RT-SFOD environment.

## Setup
- Python with Ultralytics 8.4, SAHI 0.12 and NumPy; detectron2 0.6 for the SF-UT scripts.
- Locations are read from environment variables:

| Variable | Meaning |
|---|---|
| `BASE_ROOT` | base folder with source weights (`weights/baseline_best.pt`, `yolov8n.pt`) |
| `JROOT` | working folder for data splits, caches, runs and results (default `$BASE_ROOT/work`) |
| `EVAL_CODE` | folder with the matcher and sliced-inference module `cr_eval.py` |
| `FLC_ROOT`, `IRD_ROOT` | FLC-Rural release and its YOLO-format frames and labels |
| `LLVIP_IMAGES`, `LLVIP_YAML` | LLVIP training images and data file (source models of SF-YOLO, SF-UT, YOLO26n) |
| `PYS`, `PYU` | Python of the SF-YOLO and SF-UT environments |
| `JDOMAIN` | `flc`, `forest` or `all` (all 23 recordings) |

## Order of runs
Each script lists its steps and arguments in its docstring.
1. Splits and teacher: `build_flc_splits.py`, `forest_build.py`, `forest_build_view4.py`, then `teacher_and_baseline.py`.
2. Students and caches: `b6_train.py`, `b6_epochs.py <worker> <n>`, `b9_prep.py`, `b9_train.py <n>`, `b9_epochs.py <worker> <n>`.
3. Policies: `b6_policy.py`, `b9_quick.py`, `b9_after.py`, `b9_policy.py`.
4. RT-SFOD: `b7_setup.sh`, `b7_prep.py`, `b7_train.py`, `b7_epochs.py`, `b7_policy.py`.
5. SF-YOLO and SF-UT: `b10_sfyolo.py` and `b10_sfut.py` with the steps `source`, `adapt`, `cache`, `policy` (SF-YOLO also `tam`; SF-UT also `prep`).
6. Other settings: `b10_ext_build.py` then `b10_ext_pipeline.py prep|train|epochs|policy`; `b12_sources.py prep|train|epochs|policy`; `b14_mot_build.py` then `b14_mot.py`; `b16_cityscapes.py`.

## Data
Not included. Public sets (CAMEL, BIRDSAI, MOTChallenge, Cityscapes) come from their authors under their licences; the rural recordings are on Zenodo, https://doi.org/10.5281/zenodo.22744783.

## Licence
Not set yet.
