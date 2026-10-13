# Step 4: YOLO training

Part of the [pipeline](../README.md), which explains how to run the scripts.

## `main_train_detector.py`

Trains a YOLO detector with Ultralytics from the `training` section of `configs/training.json`.

- **Settings:** the dataset yaml, GPU, W&B project / group, YOLO version and size, the optional P2 head (an extra small-object detection layer, from `p2_model_yaml`) and the training arguments.
- **Pretrained weights:** with `load_pretrained`, the standard model's pretrained weights are loaded, also into the P2 architecture (backbone and P3-P5 heads; the P2 layers start random).
- **Output:** runs go to `output/training/<wandb_project>/<run name>/`, named like the W&B run, e.g. `yolo_s_12_base` (`_2`, `_3`, ... if the name is taken).

### Evaluating a trained model

Set a model's `model_path` in the `models` of `configs/detection.json` to the run name (e.g. `"model_path": "yolo_s_12_base"`) instead of a path. It resolves to that run's `weights/best.pt`. Then run [step 5](../step5_evaluation/README.md).
