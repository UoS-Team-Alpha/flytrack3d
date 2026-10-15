# Step 5: evaluation

Part of the [pipeline](../README.md), which explains how to run the scripts.

## `main_evaluate2d.py`

Runs the enabled detectors on the detection source (`configs/shared/detection_source.json`) and scores them against the ground-truth labels. The models are the `models` of `configs/detection.json`.

- **Metrics:** AP@0.2 / 0.5 / 0.5:0.95, P/R/F1, and a side-by-side model comparison. AP is reported as "AP" when the ground truth has one class and as "mAP" (the mean over classes) when it has more.
- **Optional:** track plots and annotated videos.
- **Cache:** predictions are cached and reused while the model and frames are unchanged.
- **Batch source:** with a batch source (`shared/detection_source.batch.example.json`), each camera video becomes one input of the run, and that run is what the 3D scripts (steps 7 to 9) work on.
