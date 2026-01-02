# ADHD-EEG Signal Filtering w/ Salience Maps

This repository hosts all source code used in a project for the COS322 Foundations of Machine Learning course at the University of Southern Maine. 

To gain familiarity with neural networks while exploring a unique approach, we study how filtering EEG signal data for ADHD patients affects binary classification.

## Getting started

To reproduce the experiments conducted for this project, you'll need an up-to-date, working version of Python. 

To utilize this repository, first clone:

`git clone https://github.com/CDurepos/eeg-salience-filter`

`cd eeg-salience-filter`

Install dependencies: 

`pip install -r requirements.txt`

And finally, run the setup script:

`bash bin/setup.sh`

## Running the pipeline

This repository provides a small CLI orchestrator and standalone scripts to run each pipeline stage. Quick steps to reproduce the full pipeline locally:

Install dependencies:

```bash
pip install -r requirements.txt
bash bin/setup.sh
```

Run the full end-to-end pipeline (preprocessing, splitting, training, evaluation):

```bash
python src/run.py all

and if you do not have a data/adhdata.cav then run the following and specify the paths

python src/run.py all --data-path data/adhdata.csv --out-dir outputs/run1 --model-script teacherModels/eeg_teacher.py
```

Or run steps individually:

```bash
python src/run.py preprocess    # run preprocessing / feature extraction
python src/run.py split         # create subject splits and save epochs_subjectsplit.npz
python src/run.py train         # train models (subcommands/flags select model)
python src/run.py evaluate      # evaluate trained model(s) and save metrics
python src/run.py predict       # run inference on new input files
```

Example predict usage (adjust flags to your model/input):

```bash
python src/run.py predict --model resnet --input path/to/data.npy --output preds.json
```

Notes:

The preprocessing step produces `outputs/epochs_subjectsplit.npz` (train/val/test splits). Trainer scripts assume this file exists; run `python src/run.py split` or `python src/preprocess.py` to create it.
Model checkpoints, `history.json`, and `metrics.json` are written under `outputs/` by the training/evaluation flows.
If you prefer to run individual scripts directly, run `python src/preprocess.py` and then the trainer scripts in `src/` (they expect the split `.npz`).

If you'd like, I can add concrete example commands for training a specific model (EEGNet/ResNet/Transformer) and example flags to pass to `src/run.py`.

Alternitively you can always go into the individual files and run them allowing you to get the potput for these.