"""
main.py: runs the baseline and every experiment, then plots the results.

INPUT
    - The raw dataset, handled by DATA().build_dataset() in data.py (needs split() implemented)
    - BASE        : default training settings (the paper's Experiment 2)
    - EXPERIMENTS : one entry per run, listing only the settings that differ from BASE

PROCESS
    1. build_dataset() writes yolo_dataset/ (images, labels, data.yaml) once.
    2. For each entry in EXPERIMENTS (in order):
         config = BASE merged with that entry's overrides
         Model(name, config) -> train() -> evaluate() -> log()
       Runs are sequential, so total time is the sum of all trainings.
    3. After the loop, experiment_log.csv is read back and one bar chart is
       drawn per metric, with one bar per experiment.

OUTPUT
    - runs/output/experiment_log.csv    one row per experiment: settings + metrics 
                                        (uses log function from model.py)
    - runs/output/<metric>.png          one chart per metric (precision, recall,
                                        map50, map50_95, precision_person)

HOW TO USE
    Run:  python main.py
    Add an experiment:  add one line to EXPERIMENTS with only the keys that change,
                        e.g. "batch16": {"batch": 16}
    Change a default:   edit BASE (then delete runs/output/experiment_log.csv,
                        so the old and new columns don't get mixed)

NOTES
    - Re-running appends new rows to the CSV (only the latest row per experiment is
      plotted) and overwrites each experiment's runs/<name>/ folder.
    - If an experiment has no person boxes in the test set, precision_person is empty
      and that bar is skipped in its chart.
"""

from data import DATA                
from model import Model               
import pandas as pd                  
import matplotlib.pyplot as plt      

# BASE = the default training settings, taken from the paper's Experiment 2.
# Every experiment starts from these and overrides only the keys that differ.
BASE = {
    "model": "yolov8n.pt",   # nano variant (Sect. 4.4); .pt = COCO-pretrained = transfer learning
    "imgsz": 640,            # paper resizes everything to 640x640
    "epochs": 50,            # paper: 50 epochs for experiment 2
    "patience": 10,          # early stopping: stop if no improvement for 10 epochs (paper: 10)
    "batch": 32,             # paper: 32 for experiment 2
    "optimizer": "auto",     # unspecified in paper. With "auto" and <10,000 iterations (our case,
                             # ~2,800), Ultralytics picks AdamW with lr ~0.00111 and IGNORES lr0
    "lr0": 0.01,             # only takes effect when optimizer is set explicitly (e.g. "SGD")
    "freeze": 0,             # 0 = fine-tune all layers; N = freeze first N layers
    "seed": 0,               # fixed seed so runs are comparable
}

# EXPERIMENTS = one entry per run. Each entry lists ONLY the settings that differ
# from BASE; everything it doesn't mention is inherited from BASE.
EXPERIMENTS = {
    "baseline": {},          # no overrides: effectively AdamW ~0.00111, i.e. Ultralytics defaults

    # Optimizer and learning rate: the biggest unknowns
    "sgd": {"optimizer": "SGD", "lr0": 0.01},
    "adamw_1e-4": {"optimizer": "AdamW", "lr0": 0.0001},

    # Transfer learning depth: freeze the backbone (layers 0-9)
    "freeze10": {"freeze": 10},

    # Longer training; patience is raised with it, so this changes two keys together
    "epochs100": {"epochs": 100, "patience": 20},

    # Training from scratch, to check how much pretraining matters
    "scratch": {"model": "yolov8n.yaml"},   # .yaml = architecture only, random weights
}

# Build yolo_dataset/ and data.yaml once, before any training.
# This calls your split(), so it can't run until split() is implemented.
DATA().build_dataset()

# items() gives (experiment name, overrides dict) for each entry, so the loop
# runs once for "baseline", once for "sgd", and so on.
for name, overrides in EXPERIMENTS.items():

    # Merge the two dicts: BASE first, overrides second, so an experiment's
    # value replaces BASE's value for the same key. This is the one complete
    # settings dict that Model expects.
    config = {**BASE, **overrides}

    # Progress marker, so you can tell which run you are looking at in the terminal.
    print(f"=== Running experiment: {name} ===")

    # Create a Model for this run. It loads the weights named in config["model"]
    # (yolov8n.pt, or yolov8n.yaml for "scratch") and remembers the name for its folder.
    model = Model(name, config)

    model.train()      # trains and saves the weights to runs/<name>/weights/best.pt
    model.evaluate()   # scores best.pt on the test split and stores the metrics
    model.log()        # appends this run's settings and metrics as one row in experiment_log.csv

# Everything from here on runs AFTER the loop, once all experiments are logged.

# Read the log back as a table: one row per run, one column per setting or metric.
log = pd.read_csv(model.experiment_path)

# If main.py was run more than once, the CSV holds several rows for the same experiment.
# Keep only the most recent row of each, so every experiment gets exactly one bar.
log = log.drop_duplicates(subset="experiment", keep="last")

METRICS = list(model.metrics)

# One iteration per metric, so each metric gets its own chart.
for metric in METRICS:

    plt.figure(figsize=(8, 5))                # start a new, empty chart (8 x 5 inches)
    plt.bar(log["experiment"], log[metric])   # x = experiment names, bar height = this metric's value
    plt.title(metric)                         # chart title = metric name, e.g. "map50"
    plt.ylim(0, 1)                            # all metrics are fractions from 0 to 1, so every chart shares the same scale
    plt.xticks(rotation=45, ha="right")       # tilt the experiment names so they don't overlap
    plt.tight_layout()                        # shrink margins so the tilted labels aren't cut off

    plt.savefig(model.output_dir / f"{metric}.png", dpi=150)

# If precision_person is empty for a run 
# (no person boxes in the test set), 
# pandas reads it as NaN and plt.bar skips that bar without error.

plt.show()