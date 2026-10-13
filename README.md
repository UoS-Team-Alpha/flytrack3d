# evaluate_flydetect

Tools for evaluating fly detectors (YOLO, optionally with SAHI slicing) and turning their 2D detections from a calibrated multi-camera rig into 3D fly tracks.

- **2D:** run detectors on videos or image folders, cache the predictions, and score them against ground-truth labels: COCO-style mAP, precision / recall / F1, track plots and annotated videos.
- **3D:** match detections across cameras, triangulate them, and refine them into smooth 3D tracks with a factor graph (gtsam). The factor graph combines reprojection factors with a constant-velocity motion prior, so it bridges frames where cameras miss the fly. Tracks can also be guided by hand-annotated start / end points. They are evaluated against the ground truth in 3D (RMSE in mm) and in 2D, by projecting the tracks back into each camera.

## Setup

```bash
pip install -r requirements.txt
cp config_examples/config.example.json config.json              # single dataset
cp config_examples/config.batch.example.json config.batch.json  # or: a folder of videos + a folder of label folders
```

Fill in your paths, models and camera calibration, then delete the `"_example"` line. The scripts refuse to run a config that still has it. Every config section has a `_readme` entry that explains its settings. Relative paths in a config are resolved against the folder the config is in.

## Scripts

Every `main_*.py` script can be run from the IDE (edit its `SETTINGS` block and press Run) or from the command line, where flags override those settings (`--help` lists them). Each writes to a timestamped `output/run_<datetime>/` folder, or into the run folder it works on.

- **`main_train_detector.py`**: trains a YOLO detector with Ultralytics from the `training` section of the config: the dataset yaml, GPU, W&B project / group, YOLO version and size, the optional P2 head (an extra small-object detection layer, from `p2_model_yaml`) and the training arguments. With `load_pretrained`, the standard model's pretrained weights are loaded, also into the P2 architecture (backbone and P3-P5 heads; the P2 layers start random). Runs go to `output/training/<wandb_project>/<run name>/`, named like the W&B run, e.g. `yolo_s_12_base` (`_2`, `_3`, ... if the name is taken). To evaluate a trained model, set a model's `model_path` in `models` to the run name (e.g. `"model_path": "yolo_s_12_base"`) instead of a path; it resolves to that run's `weights/best.pt`.
- **`main_convert_dataset_format.py`**: converts a detection dataset between the YOLO and COCO formats. `yolo_to_coco` (FiftyOne) turns a YOLO dataset folder (`data.yaml`, `images/`, `labels/`) into `data/` plus `annotations/instances_<split>.json`; `coco_to_yolo` (Ultralytics) turns that layout back into `images/<split>/`, `labels/<split>/` and a `data.yaml` ready for training. The input is never changed, and `--symlink` links the images instead of copying them. It takes no config and writes to the `--output` folder.
- **`main_slice_videos.py`**: cuts the same time window (`start` to `end`) out of every video in a folder, using an ffmpeg stream copy (fast, no re-encoding; needs `ffmpeg` on the PATH). The folder and the window come from its own config, `video_config.json` (template: `config_examples/video_config.example.json`). The slices keep the videos' file names and go to `<videos_dir>/sliced` unless `output_dir` is set; existing slices are skipped.
- **`main_evaluate2d.py`**: runs the enabled detectors on the dataset in the config and scores them against the ground-truth labels: mAP@0.2 / 0.5 / 0.5:0.95, P/R/F1, a side-by-side model comparison, plus optional track plots and annotated videos. Predictions are cached and reused while the model and frames are unchanged. With a batch config, each camera video becomes one input of the run, and that run is what the 3D scripts work on.
- **`main_inference2d.py`**: runs the detectors without ground truth, on new recordings (`--inputs path/to/videos`). It writes a run folder with the same layout as an evaluation run, so the 3D scripts can use it directly.
- **`main_points3d.py`**: matches each frame's detections across the cameras (epipolar distance, Hungarian method) and triangulates them into 3D points, with no tracking. Useful for checking the calibration and the cross-camera matching.
- **`main_tracks3d.py`**: builds 3D tracks automatically. Triangulated points are linked into tracklets, each tracklet seeds a factor-graph track, and every 2D detection is associated with tracks iteratively, including detections seen by one camera only. Detections at fixed pixel positions (background clutter) are removed first. Results go to `<run>/3d/<model>/factor_graph/tracks_<datetime>/`: tracks (`tracks_3d.csv`), settings (`params.json`), 3D and reprojection plots, and, when the run has labels, `evaluation_3d.txt` / `evaluation_2d.txt`.
- **`main_tracks3d_guided.py`**: 3D tracks for hand-picked flies. An annotator opens each camera's video, where you click where each fly starts and / or ends. Later views can suggest the points from earlier views using the epipolar geometry. Each annotated fly gets exactly one track: the clicks fix its frame range and act as trusted constraints. Annotations are saved and can be reused with `--no-gui`. The annotator keeps the views in sync using each camera's `frame_offset`. Switching views opens the same instant, `g` jumps to the selected fly's point in another view, and if the same START / END is annotated in two views, the panel shows the offset that would line them up. To find an unknown offset, annotate one sharp event, like a take-off, in every view. Outputs and evaluation are the same as for `main_tracks3d.py`, in `factor_graph/guided_<datetime>/`.

Use `--source gt` on the 3D scripts to run them on the ground-truth labels instead of the predictions.

The 3D scripts also draw the red fake fruit, the green leaves around it and the arena frame, with no annotation needed. The fruit and leaves come from each view's median frame (frames sampled across the video, so the flies vanish), segmented in HSV:

- **Fruit:** round, saturated red blobs are candidates. The combination of one blob per view that triangulates consistently is the fruit, which gives the centre of its base on the wall. Its radius is measured from the blobs. It is drawn as a shaded hemisphere bulging out from the wall, towards the cameras.
- **Leaves:** the green region around the fruit in each view is laid flat onto the fruit's wall (camera rays meet the wall plane), and kept where at least two views agree. The terminal prints how well each view agrees (IoU), a quick check of the calibration.
- **Arena frame:** the tapered box of pipes is grown out from the fruit using its measured size (`arena_plotting.arena`): the back face on the fruit's wall, upright, at a fixed offset from the fruit, and the front face `depth` towards the cameras. It is only as accurate as the fruit's placement on the back wall, so check the dashed frame in the reprojection images.

The fruit and leaves are fitted once per run and cached in `<run>/3d/fruit/` and `<run>/3d/leaves/`, each with one debug image per camera (`hsv/<camera>.jpg`). The settings are in `arena_plotting` in the config. The frame needs no fitting. Use `--no-arena` to skip all three.

## Layout

- **`dataset_tools/`**: dataset creation, cleaning and conversion. `image_tools/conversion/` holds the YOLO -> COCO (`convert_yolo_to_coco.py`) and COCO -> YOLO (`convert_coco_to_yolo.py`) converters; `video_tools/slice_videos.py` cuts a time window out of a folder of videos.
- **`training/`**: building the YOLO model (version, size, P2 head, pretrained weights) and running Ultralytics training.
- **`evaluate2d/`**: datasets, detectors, metrics, prediction caching and 2D visualisation.
- **`tracking3d/`**: camera model and triangulation, run / camera loading, the factor graph (unguided and guided), evaluation, reports and 3D plots.
- **`gui/`**: the start / end annotator and its epipolar suggestions.
- **`arena_plotting/`**: static arena objects for the 3D plots. `fruit/` and `leaves/` each have `hsv/` (segmentation), `mesh/` (reconstruction and drawing) and `locate.py` (per-run glue and cache); `arena/` has `mesh/` and `locate.py` only, since nothing is segmented. `objects.py` bundles them for the plots, and `geometry.py`, `drawing.py`, `frames.py` and `settings.py` hold the code they share.
