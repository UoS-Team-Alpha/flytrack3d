# evaluate_flydetect

Tools for evaluating fly detectors (YOLO, optionally with SAHI slicing) and turning their 2D detections from a calibrated multi-camera rig into 3D fly tracks.

- **2D:** run detectors on videos or image folders, cache the predictions, and score them against ground-truth labels: COCO-style mAP, precision / recall / F1, track plots and annotated videos.
- **3D:** match detections across cameras, triangulate them, and refine them into smooth 3D tracks with a factor graph (gtsam). The factor graph combines reprojection factors with a constant-velocity motion prior, so it bridges frames where cameras miss the fly. Tracks can also be guided by hand-annotated start / end points. They are evaluated against the ground truth in 3D (RMSE in mm) and in 2D, by projecting the tracks back into each camera.

## Setup

```bash
pip install -r requirements.txt
cp config.example.json config.json              # single dataset
cp config.batch.example.json config.batch.json  # or: a folder of videos + a folder of label folders
```

Fill in your paths, models and camera calibration, then delete the `"_example"` line. The scripts refuse to run a config that still has it. Every config section has a `_readme` entry that explains its settings.

## Scripts

Every `main_*.py` script can be run from the IDE (edit its `SETTINGS` block and press Run) or from the command line, where flags override those settings (`--help` lists them). Each writes to a timestamped `output/run_<datetime>/` folder, or into the run folder it works on.

- **`main_evaluate2d.py`**: runs the enabled detectors on the dataset in the config and scores them against the ground-truth labels: mAP@0.2 / 0.5 / 0.5:0.95, P/R/F1, a side-by-side model comparison, plus optional track plots and annotated videos. Predictions are cached and reused while the model and frames are unchanged. With a batch config, each camera video becomes one input of the run, and that run is what the 3D scripts work on.
- **`main_inference2d.py`**: runs the detectors without ground truth, on new recordings (`--inputs path/to/videos`). It writes a run folder with the same layout as an evaluation run, so the 3D scripts can use it directly.
- **`main_points3d.py`**: matches each frame's detections across the cameras (epipolar distance, Hungarian method) and triangulates them into 3D points, with no tracking. Useful for checking the calibration and the cross-camera matching.
- **`main_tracks3d.py`**: builds 3D tracks automatically. Triangulated points are linked into tracklets, each tracklet seeds a factor-graph track, and every 2D detection is associated with tracks iteratively, including detections seen by one camera only. Detections at fixed pixel positions (background clutter) are removed first. Results go to `<run>/3d/<model>/factor_graph/tracks_<datetime>/`: tracks (`tracks_3d.csv`), settings (`params.json`), 3D and reprojection plots, and, when the run has labels, `evaluation_3d.txt` / `evaluation_2d.txt`.
- **`main_tracks3d_guided.py`**: 3D tracks for hand-picked flies. An annotator opens each camera's video, where you click where each fly starts and / or ends. Later views can suggest the points from earlier views using the epipolar geometry. Each annotated fly gets exactly one track: the clicks fix its frame range and act as trusted constraints. Annotations are saved and can be reused with `--no-gui`. Outputs and evaluation are the same as for `main_tracks3d.py`, in `factor_graph/guided_<datetime>/`.

Use `--source gt` on the 3D scripts to run them on the ground-truth labels instead of the predictions.

## Layout

- **`evaluate2d/`**: datasets, detectors, metrics, prediction caching and 2D visualisation.
- **`tracking3d/`**: camera model and triangulation, run / camera loading, the factor graph (unguided and guided), evaluation, reports and 3D plots.
- **`gui/`**: the start / end annotator and its epipolar suggestions.
