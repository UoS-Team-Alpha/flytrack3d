# Step 6: inference

Part of the [pipeline](../README.md), which explains how to run the scripts.

## `main_inference2d.py`

Runs the detectors without ground truth, on new recordings (`--inputs path/to/videos`). The models are the `models` of `configs/detection.json`.

- **Output:** a run folder with the same layout as an evaluation run ([step 5](../step5_evaluation/README.md)), so the 3D scripts (steps 7 to 9) can use it directly.
