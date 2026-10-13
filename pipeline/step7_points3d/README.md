# Step 7: 3D points

Part of the [pipeline](../README.md), which explains how to run the scripts. The `--source gt` flag and the fruit / leaves / arena overlays are described [there](../README.md#shared-by-the-3d-scripts-steps-7-to-9).

## `main_points3d.py`

Matches each frame's detections across the cameras (epipolar distance, Hungarian method) and triangulates them into 3D points, with no tracking. Useful for checking the calibration and the cross-camera matching.

- **Input:** the run folder that step 5 or 6 wrote.
- **Settings:** `configs/tracking.json`.
