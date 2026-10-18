# Reproducing "Achieving low-latency edge computing: YOLOv8 for real-time passenger monitoring"

A reproduction attempt of the paper's **Experiment 1**: YOLOv8-nano trained on 185 ferry-terminal images to detect 5 classes (bus, car, ferryboat, person, truck). The paper does not state the optimizer, the learning rate or the confidence threshold behind its person-precision figure, so this project tests each unknown as its own experiment. Every run is trained, scored on the test split, timed, and logged to one CSV. **Result: the paper's numbers were not reproduced** (see Results).

## Results

All numbers are on the test split (10% of the 185 images), logged in `runs/output/experiment_log.csv`. Precision and recall are averaged over the 5 classes. Each experiment appears twice: the PyTorch model (`.pt`) and its ONNX export (`_onnx`). Each was run once (seed 0).

| Experiment | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|---|---|---|---|---|
| **Paper, Experiment 1 (target)** | **65.6** | **42.2** | **43.0** | **32.7** |
| baseline | 32.4 | 27.6 | 25.2 | 9.4 |
| baseline_onnx | 29.9 | 37.5 | 27.0 | 10.4 |
| sgd | 42.5 | 25.4 | 15.5 | 6.1 |
| sgd_onnx | 44.7 | 20.0 | 16.3 | 7.4 |
| **adamw_1e-4** | 41.7 | 47.6 | 41.4 | 19.2 |
| **adamw_1e-4_onnx** | 47.4 | 53.4 | 44.5 | 21.0 |
| freeze10 | 34.7 | 36.4 | 26.3 | 10.2 |
| freeze10_onnx | 18.7 | 54.5 | 26.1 | 10.0 |
| scratch | 53.0 | 7.2 | 5.3 | 2.2 |
| scratch_onnx | 48.4 | 8.1 | 5.7 | 2.1 |

All values in %.

**Takeaway**
- **`adamw_1e-4` came closest** (average gap to the four targets: 11.1 points for `.pt`, 10.6 for ONNX). Its mAP@0.5 (41.4 for `.pt`, 44.5 for ONNX) is within 2 points of the paper's 43.0, and its recall is above the paper's.
- It is **still not a reproduction**. Precision is 18 to 24 points below the paper's 65.6, and mAP@0.5-0.95 is 12 to 14 points below the paper's 32.7. The paper's mAP@0.5-0.95 is 76% of its mAP@0.5, while here it is 46%, which suggests looser box localization.
- In these runs, `baseline` (Ultralytics "auto", which picks AdamW at about 0.00111), `sgd` and `freeze10` stayed far below the paper on mAP. Training from scratch (`scratch`) nearly fails (recall about 7%).
- ONNX scores differ from the `.pt` scores, most visibly for `freeze10` (precision 34.7 to 18.7, recall 36.4 to 54.5). The cause was not investigated.
- With one seed per experiment, small differences between runs are not conclusive.

### Person precision

The paper's headline 97% person precision comes from its **Experiment 2** (2,229 images). Experiment 1 reports no person-specific number, so there is **no paper target** for this table. `precision_person` is measured at the best-F1 confidence; the other columns are at fixed confidence thresholds.

| Experiment | Best-F1 | @0.25 | @0.5 | @0.7 | @0.9 |
|---|---|---|---|---|---|
| baseline | 57.0 | 53.3 | 100 | 100 | 100 |
| baseline_onnx | 68.0 | 75.7 | 93.9 | 100 | 100 |
| sgd | 100 | 100 | 100 | 100 | 100 |
| sgd_onnx | 100 | 100 | 100 | 100 | 100 |
| adamw_1e-4 | 100 | 100 | 100 | 100 | 100 |
| adamw_1e-4_onnx | 100 | 100 | 100 | 100 | 100 |
| freeze10 | 100 | 6.6 | 100 | 100 | 100 |
| freeze10_onnx | 31.6 | 8.0 | 41.8 | 100 | 100 |
| scratch | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| scratch_onnx | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

All values in %. A 100 can rest on very few detections, since the log does not record detection counts. A 0 means no correct person detection.

### Latency (measured, Google Colab T4 GPU)

All logged latency was measured on a **Google Colab T4 GPU**, not on an edge device. Milliseconds per image, batch 1, 640x640, mean of 1,000 runs after 100 warm-up runs.

| Experiment | Preprocess | Inference | Postprocess | Total |
|---|---|---|---|---|
| baseline | 0.69 | 6.23 | 1.14 | 8.06 |
| baseline_onnx | 0.68 | 6.00 | 1.21 | 7.89 |
| sgd | 0.69 | 6.32 | 1.16 | 8.18 |
| sgd_onnx | 0.69 | 6.04 | 1.25 | 7.98 |
| adamw_1e-4 | 0.71 | 6.35 | 1.15 | 8.20 |
| adamw_1e-4_onnx | 0.69 | 6.11 | 1.21 | 8.01 |
| freeze10 | 0.70 | 6.42 | 1.20 | 8.31 |
| freeze10_onnx | 0.67 | 6.11 | 1.25 | 8.04 |
| scratch | 0.70 | 6.20 | 0.49 | 7.39 |
| scratch_onnx | 0.67 | 5.94 | 0.54 | 7.15 |

For comparison, the paper reports **5.7 ms** total for Experiment 1 on a Colab Pro T4 (0.2 preprocess + 4.2 inference + 1.3 postprocess). The ONNX runs here are about 7.9 to 8.0 ms, mainly because preprocessing and inference are slower. The reason was not investigated.

## Project structure

```
model.py            Model class: train, evaluate, ONNX/TFLite export, latency, log
main.py             runs every experiment, then plots one bar chart per metric
requirements.txt    ultralytics, pyyaml, pandas, matplotlib, onnxruntime
adam_optimizer.png  Edge Impulse profile of adamw_1e-4 best.onnx on Raspberry Pi 5
yolo_dataset/       NOT in the repo (private); you add it, see Dataset
runs/               created when you run main.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is **not in this repository**. It is Experiment 1 of the paper (185 images, 80/10/10 split) and was provided privately by the paper's authors, **Roberto de Pádua Carvalho Reis** and **Omar Andres Carmona Cortes**. It is already in YOLO format. Place it, with the authors' `data.yaml`, like this:

```
yolo_dataset/
├── data.yaml
├── train/  images/  labels/
├── val/    images/  labels/
└── test/   images/  labels/
```

Class ids: bus=0, car=1, ferryboat=2, person=3, truck=4.

## Evaluation metrics

- **IoU (Intersection over Union):** how well a predicted box overlaps the true box, as overlap area ÷ combined area. It goes from 0 (no overlap) to 1 (identical).
- **Correct detection:** the class is right and the IoU is above a chosen threshold.
- **Precision:** of the boxes the model predicted, the fraction that were correct.
- **Recall:** of the real objects in the images, the fraction the model found.

**Average Precision (AP), for one class**

1. The model gives every predicted box a confidence score.
2. Sweep the confidence cutoff from high to low. At each cutoff, keep only the boxes at or above it, and calculate precision and recall.
3. A high cutoff keeps only the surest boxes, so precision is high and recall is low. A low cutoff keeps more boxes, so recall rises and precision usually falls.
4. Plot recall on the x-axis and precision on the y-axis. The result is the precision-recall curve.
5. AP is the area under that curve, which is the same as the average of the precision values across all recall levels. It is one number from 0 to 1.

A model that holds high precision all the way to high recall has a curve that stays near the top, so its area is close to 1. A model whose precision collapses as recall rises has a curve that drops fast and a small area.

- **mAP:** the average of AP over all classes.

**mAP50 vs mAP50-95**

- **mAP50:** a box counts as correct if IoU ≥ 0.5, meaning it only has to overlap the true box by half. It is lenient and tells you whether the model finds the objects.
- **mAP50-95:** the average of mAP at IoU thresholds 0.50, 0.55, 0.60 and so on up to 0.95. The high thresholds require tight boxes, so this one is strict and tells you how precisely the model localizes. It is always lower than mAP50.

**Also logged**
- **Person precision:** precision for the `person` class only, at the best-F1 confidence and at fixed confidence thresholds (0.25, 0.5, 0.7, 0.9). A confidence threshold is a cutoff: boxes below it are discarded. A higher threshold usually raises precision and lowers recall.
- **Latency:** time per image from image in to boxes out, split into preprocess, inference and postprocess (total = the sum).

## How to run

```bash
python main.py
```

For each experiment, `main.py` trains, evaluates the `.pt` model and its ONNX export on the test split, times both, and appends two rows to the log. Experiments run one after another, so total time is the sum of all trainings. Afterwards it draws one bar chart per metric.

Outputs:
- `runs/<name>/`: weights (`best.pt`, `best.onnx`) and training curves
- `runs/<name>_test/` and `runs/<name>_onnx_test/`: test plots for the `.pt` and ONNX models
- `runs/output/experiment_log.csv`: two rows per experiment (`<name>` and `<name>_onnx`)
- `runs/output/<metric>.png`: one bar chart per metric

To add an experiment, add one line to `EXPERIMENTS` in `main.py` with only the settings that change, for example `"batch32": {"batch": 32}`.

## Experiments explained

`BASE` in `main.py` is the paper's Experiment 1 setup: `yolov8n.pt`, 640x640, 200 epochs, batch 16, patience 10 (a guess), seed 0. Each experiment changes only what is listed.

- `baseline`: no changes. The optimizer is `"auto"`, which picks AdamW (about 0.00111) and ignores `lr0`.
- `sgd`: SGD optimizer, `lr0` 0.01.
- `adamw_1e-4`: AdamW optimizer, `lr0` 0.0001.
- `freeze10`: freezes the first 10 layers (the backbone) and fine-tunes the rest.
- `scratch`: `yolov8n.yaml`, so the same architecture with random weights and no pretraining.

The confidence threshold is not a separate training run. It is covered by the person-precision columns at several thresholds.

## Raspberry Pi 5 deployment

**These are Edge Impulse ESTIMATES, not measurements, and they are not comparable to the T4 latency above.**

- Profiled model: `best.onnx` exported from the **`adamw_1e-4`** experiment, input shape (640, 640, 3)
- Target: Raspberry Pi 5; engine shown by Edge Impulse: TensorFlow Lite
- Estimated inferencing time: **244 ms**
- Flash usage: 12.1M. Edge Impulse notes the model won't run on MCUs (arena size >6MB).

<img width="1097" height="546" alt="adam_optimizer" src="https://github.com/user-attachments/assets/df2de272-da5d-431b-a013-66e2daecc290" />

The paper does not cover the Pi 5. For context, the Edge Impulse estimates for the two boards the paper does report are Raspberry Pi 4: 4160 ms and Jetson Nano: 2985 ms. The paper reports **1790 ms** (Pi 4B) and **5.3 ms** (Jetson Nano), measured with ONNX Runtime. Edge Impulse used a different engine (TensorFlow Lite) and only estimates, so these figures should not be read as a like-for-like comparison.

## Known limitations

- **Not a successful reproduction.** No run matches all four Experiment 1 targets (see Results).
- **Paper details missing:** the optimizer, learning rate and the confidence threshold for person precision are unspecified. The paper also does not say whether its Table 3 metrics are on the validation or the test split; this project uses the test split. For Experiment 1 the paper gives no patience value, only that early stopping ended training at epoch 47. `patience=10` here is a guess (the paper gives it for Experiment 2).
- **Only one seed per experiment**, so small gaps are not conclusive.
- The paper's 97% person precision belongs to Experiment 2, which is not reproduced here.
- `latency()` measures on the machine it runs on (a Colab T4 here), not on the paper's Jetson Nano. The `.pt` rows time PyTorch, and the `_onnx` rows time ONNX Runtime.
- `"auto"` picks AdamW (about 0.00111) and ignores `lr0` at this dataset size.
- Delete `runs/output/experiment_log.csv` whenever the keys in `BASE` or the logged metrics change, or the columns will misalign.
- The `epochs100` experiment is commented out in `main.py` and was not run.
- The Edge Impulse numbers are estimates only.
- The dataset is private and not included.

## Reference

Reis, R. de P. C., Cortes, O. A. C. (2026). *Achieving low-latency edge computing: a performance analysis of YOLOv8 for real-time passenger monitoring.* Computing 108:119. https://doi.org/10.1007/s00607-026-01704-2
