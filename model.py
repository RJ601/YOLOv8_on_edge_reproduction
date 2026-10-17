"""
model.py: trains, tests and logs one YOLOv8 experiment.

INPUT
    - yolo_dataset/data.yaml, written by build_dataset() in the dataset script
      (it points to the train/val/test image folders and holds the class names)
    - name   : experiment name, e.g. "baseline" or "sgd"
    - config : one complete settings dict (BASE merged with that experiment's overrides)

PROCESS (one Model object = one experiment, three steps in this order)
    1. train()    : fine-tunes YOLOv8 on the train split with the given settings (config);
                    the val split is used during training to pick the best epoch
    2. evaluate() : loads the best checkpoint (best.pt) and scores it on the test
                    split, which training never saw
    3. log()      : appends this run's settings and test metrics as one row to a
                    shared CSV

OUTPUT
    - runs/<name>/           weights (best.pt, last.pt), results.csv, training curves
    - runs/<name>_test/      plots from the test-split evaluation
    - runs/output/experiment_log.csv   one row per experiment: all settings plus
                             precision, recall, map50, map50_95, precision_person
    - evaluate() also returns the metrics dict (values between 0 and 1)

HOW TO USE e.g. (this is what main.py does)
    from model import Model

    config = {**BASE, **EXPERIMENTS["sgd"]}   # complete settings for this run
    m = Model("sgd", config)
    m.train()
    m.evaluate()
    m.log()

NOTES
    - log() appends, so re-running an experiment adds a second row. If you add or
      remove a key in BASE, delete experiment_log.csv first, or the columns won't line up.
    - Re-running an experiment overwrites its runs/<name>/ folder.
"""

from pathlib import Path        # OS-independent way to build file paths
from ultralytics import YOLO    # the YOLOv8 class: loads weights, trains, validates (pip install ultralytics)
import csv

class Model:
    def __init__(self, name, config):
        """
        name   : experiment name, e.g. "baseline" or "sgd"; also used as the run folder name
        config : the full settings dict for this run (BASE already merged with
                 the experiment's overrides, e.g. {**BASE, **EXPERIMENTS["sgd"]})
        """
        self.name = name        # remembered so train() and log() can label their output
        self.config = config    # remembered so train() can pass these settings to Ultralytics

        # Load the starting weights. "yolov8n.pt" = COCO-pretrained (transfer learning),
        # "yolov8n.yaml" = architecture only with random weights (the "scratch" experiment).
        self.model = YOLO(config["model"])

        # Will hold the evaluation metrics once evaluate() has run; None means "not evaluated yet".
        self.metrics = None

        # Folder where every training run is saved (weights, curves, results.csv).
        self.RUNS_DIR = Path("runs")

        # data.yaml written by your dataset script (build_dataset). Ultralytics reads
        # the split folders and the class names from this file.
        self.DATA_YAML = Path("yolo_dataset") / "data.yaml"

        # Class id of "person" in the CLASSES list (bus=0, car=1, ferryboat=2, person=3, truck=4).
        # The paper's headline 97% precision is for this class only, so we'll need its id later.
        self.PERSON_CLASS_ID = 3

        self.output_dir = self.RUNS_DIR / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)        # create runs/output/ if it doesn't exist yet
        self.experiment_path = self.output_dir / "experiment_log.csv"

    def train(self):
        """Trains the model with this run's settings. Output goes to runs/<name>/."""

        # "model" was only needed in __init__ to load the weights. train() does not
        # accept it as a setting, so copy every other key of the config into train_args.
        train_args = {k: v for k, v in self.config.items() if k != "model"}

        # data      : path to data.yaml (where the splits and class names live)
        # project   : parent folder for the run; .resolve() makes it absolute so
        #             Ultralytics saves exactly here and does not add its own subfolder
        # name      : run folder name, i.e. the experiment name (runs/baseline, runs/sgd, ...)
        # exist_ok  : re-running an experiment overwrites its folder instead of
        #             creating baseline2, baseline3, ...
        # **train_args unpacks into epochs, patience, batch, imgsz, optimizer, lr0, freeze, seed
        self.model.train(data=str(self.DATA_YAML), project=str(self.RUNS_DIR.resolve()),
                         name=self.name, exist_ok=True, **train_args)

    def evaluate(self):
        """Evaluates the best checkpoint of this run on the held-out TEST split."""

        # best.pt is the checkpoint train() saved at runs/<name>/weights/best.pt.
        # We evaluate this one (not last.pt) because it scored best on validation.
        best_path = self.RUNS_DIR.resolve() / self.name / "weights" / "best.pt"

        # Load it into a fresh YOLO object so the evaluation does not depend on
        # whatever state self.model was left in after training.
        best_model = YOLO(str(best_path))

        # .val() runs the model over a split and computes precision, recall and mAP.
        # split="test" : use images/test from data.yaml, which training never saw
        #                (the default would be the val split used during training)
        # imgsz        : same 640 as training, so the settings are consistent
        # project/name : saves the evaluation plots in runs/<name>_test/ instead of
        #                Ultralytics' default folder, so nothing gets mixed up
        results = best_model.val(data=str(self.DATA_YAML), split="test",
                                 imgsz=self.config["imgsz"],
                                 project=str(self.RUNS_DIR.resolve()),
                                 name=f"{self.name}_test", exist_ok=True)
        
        # results.box holds the detection metrics, averaged over all 5 classes
        box = results.box

        # Turn the numbers into plain Python floats and store them for log() to use later.
        # mp       : mean precision over classes (the paper's "Precision (Overall)")
        # mr       : mean recall over classes
        # map50    : mAP at IoU threshold 0.5
        # map      : mAP averaged over IoU 0.5 to 0.95 (the paper's "mAP@0.5-0.95")
        self.metrics = {"precision": float(box.mp), "recall": float(box.mr),
                        "map50": float(box.map50), "map50_95": float(box.map)}

        # box.ap_class_index lists the class ids that actually appear in the test set;
        # box.p holds one precision value per entry of that list, in the same order.
        present = list(box.ap_class_index)

        # Person precision = the entry of box.p at person's position in "present".
        # If the test set has no person boxes, store None instead of crashing.
        has_person = self.PERSON_CLASS_ID in present
        person_p = float(box.p[present.index(self.PERSON_CLASS_ID)]) if has_person else None
        self.metrics["precision_person"] = person_p

        # Hand the metrics back so main.py can also use them directly after the call.
        return self.metrics

    def log(self):
        """Appends one row (settings + test metrics) for this run to runs/experiment_log.csv."""

        # Logging only makes sense after evaluate() has filled self.metrics.
        # Stop with a clear message instead of writing an empty row.
        if self.metrics is None:
            raise RuntimeError("Call evaluate() before log().")

        # One row = experiment name + every setting + every metric.
        # {**a, **b} merges dicts, so each key becomes one column of the table.
        row = {"experiment": self.name, **self.config, **self.metrics}

        # The one shared log file that every experiment appends to.
        # mkdir makes sure the runs/ folder exists even if this is the very first call.
        log_path = self.experiment_path
        self.RUNS_DIR.mkdir(parents=True, exist_ok=True)

        # The header row (column names) must be written only once, on the very first run.
        # Check BEFORE opening the file, because opening in append mode creates it.
        is_new_file = not log_path.exists()

        # "a" = append: every run adds a row below the previous ones, nothing is overwritten.
        # newline="" stops Windows from inserting a blank line between rows.
        with open(log_path, "a", newline="") as f:
            # fieldnames = the column names, taken from the keys of row (same order as built above)
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new_file:
                writer.writeheader()   # column names, first run only
            writer.writerow(row)       # this run's settings and metrics

        print(f"Logged '{self.name}' to {log_path}")