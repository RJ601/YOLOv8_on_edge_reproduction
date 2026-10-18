"""
model.py: trains, tests, times and logs one YOLOv8 experiment (.pt model and its ONNX export).

INPUT
    - data.yaml (the dataset's yaml; it points to the train/val/test folders and holds
      the class names). Its path is DATA_YAML in __init__.
    - name   : experiment name, e.g. "baseline" or "sgd"
    - config : one complete settings dict (BASE merged with that experiment's overrides)

PROCESS (one Model object = one experiment, four steps in this order)
    1. train()         : fine-tunes YOLOv8 on the train split; the val split picks the best epoch
    2. evaluate()      : scores best.pt on the test split and times it (latency)
    3. evaluate_onnx() : exports best.pt to ONNX, then scores and times the ONNX model
                         the same way, to show what is lost when deploying on the edge
    4. log()           : appends one row for the .pt model and one for the ONNX model

OUTPUT
    - runs/<name>/                weights (best.pt, best.onnx), training curves
    - runs/<name>_test/           test-split plots for the .pt model
    - runs/<name>_onnx_test/      test-split plots for the ONNX model
    - runs/output/experiment_log.csv   two rows per experiment: "<name>" and "<name>_onnx".
                                  Columns: settings, precision, recall, map50, map50_95,
                                  precision_person, precision_person_at_<threshold>,
                                  latency_*_ms
    - evaluate() and evaluate_onnx() also return their metrics dicts

HOW TO USE (this is what main.py does)
    from model import Model

    config = {**BASE, **EXPERIMENTS["sgd"]}   # complete settings for this run
    m = Model("sgd", config)
    m.train()
    m.evaluate()
    m.evaluate_onnx()
    m.log()
    Also available: load(), export_tflite()

NOTES
    - log() appends, so re-running an experiment adds new rows. If you add or remove a
      key in BASE or a metric, delete experiment_log.csv first, or the columns won't line up.
    - Re-running an experiment overwrites its runs/<name>/ folder.
    - Latency is measured on the machine you run on, not on the target edge device.
"""

from pathlib import Path        # OS-independent way to build file paths
from ultralytics import YOLO    # the YOLOv8 class: loads weights, trains, validates (pip install ultralytics)
import csv
import numpy as np

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
        self.onnx_metrics = None

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

    def load(self, weights_path=None):
        """
        Loads a TRAINED checkpoint into self.model, so evaluate(), export and
        latency can run without training again.

        weights_path : path to a .pt file. If None, uses this experiment's own
                       best checkpoint, runs/<name>/weights/best.pt
        """

        # No path given -> fall back to the best.pt that train() saved for this experiment.
        if weights_path is None:
            weights_path = self.RUNS_DIR.resolve() / self.name / "weights" / "best.pt"

        # Fail early with a clear message instead of a confusing error from inside Ultralytics.
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"No weights found at {weights_path}. Run train() first.")

        # Replace the model loaded in __init__ with the trained one.
        self.model = YOLO(str(weights_path))

    def person_precision_at(self, box, thresholds=(0.25, 0.5, 0.7, 0.9)):
        """
        Person precision at fixed confidence thresholds (box = results.box from evaluate()).

        WHAT IS A CONFIDENCE THRESHOLD?
        For every box it predicts, the model also gives a confidence score between 0 and 1
        (how sure it is). A threshold is a cutoff: predictions BELOW it are thrown away,
        predictions at or above it are kept.
            - high threshold (0.9): only the very sure boxes survive -> fewer false
              detections, so precision goes UP, but real people get missed (recall goes DOWN)
            - low threshold (0.25): many boxes survive -> more people found, but more
              false detections, so precision goes DOWN
        The paper's 97% person precision only means something at ONE specific threshold,
        and the paper does not say which. So we measure several and compare.
        """

        # box.ap_class_index = the class ids that really appear in the test set.
        # Every curve array below has one row per entry of this list, in the same order.
        present = list(box.ap_class_index)

        # If the test set contains no person boxes there is nothing to measure,
        # so return an empty dict instead of crashing.
        if self.PERSON_CLASS_ID not in present:
            return {}

        # box.p_curve = precision of each class, calculated at 1000 evenly spaced
        # confidence cutoffs from 0 to 1. Picking the row at person's position
        # gives person's curve: 1000 precision values, one per cutoff.
        # (box.px holds those 1000 cutoff values; we use it in the next block.)
        curve = box.p_curve[present.index(self.PERSON_CLASS_ID)]

        # Will hold one result per threshold, e.g. {"precision_person_at_0.5": 0.91, ...}
        out = {}

        for t in thresholds:
            # np.interp(x, xs, ys) reads the curve's value at position x.
            # xs = box.px (the 1000 confidence cutoffs), ys = curve (person precision at each cutoff),
            # so this answers: "what was person precision if we keep only boxes with confidence >= t?"
            # If the model made NO person predictions that confident, Ultralytics stores 0 there,
            # so a 0 at a high threshold can mean "no detections", not "all wrong".
            out[f"precision_person_at_{t}"] = float(np.interp(t, box.px, curve))

        return out   

    def score(self, yolo_model, run_name):
        """
        Scores ANY loaded YOLO model (the .pt one or the .onnx one) on the TEST split
        and returns the metrics as a dict.

        yolo_model : a YOLO object (loaded from best.pt or from best.onnx)
        run_name   : folder name for the evaluation plots under runs/,
                     e.g. "baseline_test" or "baseline_onnx_test"
        """

        # .val() runs the model over a split and computes precision, recall and mAP.
        # split="test"  : use the test images from data.yaml, which training never saw
        # imgsz         : same 640 as training
        # project/name  : saves this evaluation's plots in runs/<run_name>/
        results = yolo_model.val(data=str(self.DATA_YAML), split="test",
                                 imgsz=self.config["imgsz"],
                                 project=str(self.RUNS_DIR.resolve()),
                                 name=run_name, exist_ok=True)

        # results.box holds the detection metrics, averaged over all classes
        box = results.box

        # mp = mean precision, mr = mean recall, map50 = mAP at IoU 0.5,
        # map = mAP averaged over IoU 0.5 to 0.95. float() makes them plain numbers.
        metrics = {"precision": float(box.mp), "recall": float(box.mr),
                   "map50": float(box.map50), "map50_95": float(box.map)}

        # box.ap_class_index = the class ids that actually appear in the test set.
        # box.p holds one precision value per entry of that list, in the same order.
        present = list(box.ap_class_index)

        # Person precision = the entry of box.p at person's position in "present".
        # If the test set has no person boxes, store None instead of crashing.
        has_person = self.PERSON_CLASS_ID in present
        metrics["precision_person"] = float(box.p[present.index(self.PERSON_CLASS_ID)]) if has_person else None

        # Add person precision at the fixed confidence thresholds (0.25, 0.5, 0.7, 0.9).
        # dict.update() merges those extra keys into metrics.
        metrics.update(self.person_precision_at(box))

        # Hand the finished metrics back to whoever called score().
        return metrics

    def latency(self, yolo_model, warmup=100, runs=1000):
        """
        Average time to process ONE image with the model, following the paper's
        protocol (Sect. 4.2.1): batch size 1, 640x640 input, 100 warm-up runs that are
        thrown away, then the average of 1,000 timed runs over test-set images.

        LATENCY
        The time one image takes from "image in" to "boxes out", in milliseconds.
        It is split into three parts, which Ultralytics measures separately:
            preprocess  : resizing and normalizing the image
            inference   : the neural network itself
            postprocess : turning raw output into final boxes (non-max suppression)
        The paper's figure is the total of the three (end-to-end).

        WARM-UP
        The first few runs are slower (memory allocation, caches, GPU start-up), so
        they are run first and NOT counted, otherwise the average would be unfairly high.
        """

        # The test images listed in the dataset folder: <dataset>/test/images/*
        test_dir = self.DATA_YAML.parent / "test" / "images"
        images = sorted(test_dir.glob("*.*"))

        # Nothing to time if the test folder is empty (e.g. DATA_YAML points to the wrong place).
        if not images:
            raise FileNotFoundError(f"No test images found in {test_dir}")

        speeds = []   # will hold one timing dict per TIMED run (warm-up runs are not stored)

        for i in range(warmup + runs):

            # i % len(images) cycles through the test images, so 1,000 runs work even if
            # the test set has fewer images. Passing ONE path = batch size 1, as in the paper.
            # imgsz keeps the 640x640 input; verbose=False stops per-image printing.
            # predict() returns a list with one Results object; [0] takes it.
            r = yolo_model.predict(str(images[i % len(images)]),
                                   imgsz=self.config["imgsz"], verbose=False)[0]

            # r.speed = {"preprocess": ms, "inference": ms, "postprocess": ms} for this image.
            # Only keep it once the warm-up phase is over.
            if i >= warmup:
                speeds.append(r.speed)

        # Average each of the three parts over all timed runs, in milliseconds.
        pre = float(np.mean([s["preprocess"] for s in speeds]))
        inf = float(np.mean([s["inference"] for s in speeds]))
        post = float(np.mean([s["postprocess"] for s in speeds]))

        # The paper's figure is the end-to-end total: all three parts added together.
        result = {"latency_preprocess_ms": pre, "latency_inference_ms": inf,
                  "latency_postprocess_ms": post, "latency_total_ms": pre + inf + post}

        return result
        
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

        self.metrics = self.score(best_model, f"{self.name}_test")
        self.metrics.update(self.latency(best_model))

        # Hand the metrics back so main.py can also use them directly after the call.
        return self.metrics

    def evaluate_onnx(self):
        """Exports best.pt to ONNX, then scores and times the ONNX model like evaluate() does."""

        # Convert runs/<name>/weights/best.pt to runs/<name>/weights/best.onnx.
        # export_onnx() returns the path of the new file.
        onnx_path = self.export_onnx()

        # Load the .onnx file. Ultralytics runs it through ONNX Runtime internally.
        # task="detect" tells it the model finds boxes, because an .onnx file does not store that.
        onnx_model = YOLO(str(onnx_path), task="detect")

        # Score the ONNX model with the same helper as the .pt model, so the numbers are
        # comparable. Plots go to runs/<name>_onnx_test/, separate from the .pt evaluation.
        self.onnx_metrics = self.score(onnx_model, f"{self.name}_onnx_test")
        self.onnx_metrics.update(self.latency(onnx_model))

        # Kept in its own dict (not self.metrics), so log() can write the .pt and
        # ONNX results as two separate rows.
        return self.onnx_metrics

    def export_onnx(self):
        """
        Converts this experiment's best checkpoint (.pt) into an ONNX file (.onnx).

        WHAT IS ONNX?
        A .pt file only runs through PyTorch/Ultralytics. ONNX is a neutral, standard
        model format that other runtimes can load. The paper measured its latency
        with ONNX Runtime
        """

        self.load()

        # Use the same input size as training (640), so the exported model expects
        # exactly the image shape it was trained on. The paper also used 640x640.
        imgsz = self.config["imgsz"]

        # .export() converts the loaded model and returns the path of the new file.
        # format="onnx" : the target format
        # imgsz         : fixed input size, baked into the file (640 x 640)
        # batch=1       : the model accepts one image at a time, as in the paper's latency protocol
        # dynamic=False : input shape is fixed, not variable, which is simpler and often faster
        path = self.model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False)

        # Ultralytics saves the file next to the checkpoint: runs/<name>/weights/best.onnx
        # Keep the path on the object so latency() can find the file later.
        self.onnx_path = Path(path)

        return self.onnx_path  

    def export_tflite(self, int8=False):
        """
        Converts this experiment's best checkpoint (.pt) into a TensorFlow Lite file (.tflite).

        WHAT IS TFLITE?
        TFLite is TensorFlow's format for small devices (phones, microcontrollers,
        Raspberry Pi). The paper did NOT use it (it used ONNX), so this is an extra.

        int8 : False = keep normal float weights (safe default).
               True  = quantize weights to 8-bit integers. Ultralytics needs
                       sample images from data.yaml to calibrate the conversion.
        """

        # Load the TRAINED weights (runs/<name>/weights/best.pt) into self.model,
        # so the export converts the result of training, not the untrained starting weights.
        self.load()

        # Same 640 input size as training, baked into the exported file.
        imgsz = self.config["imgsz"]    

        # int8            : True = 8-bit quantization (see the docstring), False = float weights
        # data            : data.yaml, so that with int8=True Ultralytics can read sample
        #                   images from it to calibrate the quantization (ignored when int8=False)
        path = self.model.export(format="tflite", imgsz=imgsz, int8=int8,
                                 data=str(self.DATA_YAML))

        # Ultralytics writes the file inside a folder next to the checkpoint, under
        # runs/<name>/weights/. Keep the path on the object so you can find it later.
        self.tflite_path = Path(path)

        return self.tflite_path

    def log(self):
            """Appends one row (settings + test metrics) for this run to runs/experiment_log.csv."""
    
            # Logging only makes sense after evaluate() has filled self.metrics.
            # Stop with a clear message instead of writing an empty row.
            if self.metrics is None:
                raise RuntimeError("Call evaluate() before log().")
    
            # One row per model: the .pt model under the experiment's name, and the ONNX
            # model (only if evaluate_onnx() ran) under "<name>_onnx".
            rows = [{"experiment": self.name, **self.config, **self.metrics}]
            if self.onnx_metrics is not None:
                rows.append({"experiment": f"{self.name}_onnx", **self.config, **self.onnx_metrics})
    
            # The one shared log file that every experiment appends to.
            # mkdir makes sure the runs/ folder exists even if this is the very first call.
            log_path = self.experiment_path
            self.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
            # The header row (column names) must be written only once, on the very first run.
            # Check BEFORE opening the file, because opening in append mode creates it.
            is_new_file = not log_path.exists()
    
            # "a" = append: nothing already in the file is overwritten. newline="" stops
            # Windows from inserting blank lines between rows.
            with open(log_path, "a", newline="") as f:
                # Column names come from the first row. extrasaction="ignore" skips a key that
                # is missing from the header instead of crashing, and restval="" fills a column
                # the row doesn't have (e.g. no person boxes -> no person-threshold columns).
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                                        extrasaction="ignore", restval="")
                if is_new_file:
                    writer.writeheader()   # column names, first run only
                writer.writerows(rows)     # one row for the .pt model, one for the ONNX model

            print(f"Logged {[r['experiment'] for r in rows]} to {log_path}")