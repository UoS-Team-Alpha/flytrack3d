# Step 9: visualise 3D

Part of the [pipeline](../README.md), which explains how to run the scripts. The `--source gt` flag and the fruit / leaves / arena overlays are described [there](../README.md#shared-by-the-3d-scripts-steps-7-to-9).

## `main_visualise_past_3d.py`

Opens the interactive 3D plot of a past step 7 / 8 result again, without recomputing it. Nothing is written to the result's folder.

- **Choosing a result:** in the terminal, choose a run, then what to view (its triangulated points, a past unguided run or a past guided run), then which one if there are several.
- **Read from the result:** the points / tracks come from the saved csv.
- **Rebuilt:** the cameras, ground-truth overlay, arena and the grey points behind the tracks are rebuilt using the settings and frame offsets saved in the result's `params.json`.
- **Calibration:** it is not saved with a result, so those parts follow the current `cameras` config.

## `main_visualise_external_points.py`

The same 3D plot for files that come without their run.

- **Config:** it has its own, `configs/external_points.json` (template: `configs/external_points.example.json`), which holds the inputs and includes the shared calibration (`cameras`) and `tracking3d` / `arena_plotting` settings.
- **Inputs:** set in the config's `inputs` section or with the flags below.

### Inputs

| Flag | Give it |
| --- | --- |
| `--csv` | a `points.csv`, a `tracks_3d.csv`, or any csv with `frame, x, y, z` columns; a `track` column makes it a track plot |
| `--labels` | YOLO labels of the detector's predictions: a folder with one sub-folder per camera, named after it |
| `--fg-labels` | YOLO labels of the boxes the factor graph inserted, in the same layout |
| `--videos` | a folder with one video per camera, needed for the arena overlays |

Give it a csv, or labels (`--labels` and / or `--fg-labels`).

- **Labels:** they are pooled, shifted by each camera's `frame_offset`, and matched and triangulated as in [step 7](../step7_points3d/README.md). `--link` also links the points into tracks (no factor graph).
- **Arena overlays:** the fruit, leaves and arena frame are drawn only with `--videos`, since they are found in the images. Their fits are cached in `output/external/<videos folder>/3d/`.
- **Saving:** `--save plot.png` also saves the figure.
