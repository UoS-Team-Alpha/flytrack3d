# evaluate_flydetect

Tools for evaluating fly detectors (YOLO, optionally with SAHI slicing) and turning their 2D detections from a calibrated multi-camera rig into 3D fly tracks.

- **2D:** run detectors on videos or image folders, cache the predictions, and score them against ground-truth labels: COCO-style mAP, precision / recall / F1, track plots and annotated videos.
- **3D:** match detections across cameras, triangulate them, and refine them into smooth 3D tracks with a factor graph (gtsam). The factor graph combines reprojection factors with a constant-velocity motion prior, so it bridges frames where cameras miss the fly. Tracks can also be guided by hand-annotated start / end points. They are evaluated against the ground truth in 3D (RMSE in mm) and in 2D, by projecting the tracks back into each camera.

## Setup

```bash
pip install -r requirements.txt
cp config_examples/config.example.json config.json              # single dataset
cp config_examples/config.batch.example.json config.batch.json  # or: a folder of videos + a folder of label folders
cp config_examples/video_config.example.json video_config.json                  # step 0: video editing
cp config_examples/image_dataset_config.example.json image_dataset_config.json  # step 1: image dataset creation
cp config_examples/image_tiling_config.example.json image_tiling_config.json    # step 2: image tiling
```

Fill in your paths, models and camera calibration, then delete the `"_example"` line. The scripts refuse to run a config that still has it. Every config section has a `_readme` entry that explains its settings. Relative paths in a config are resolved against the folder the config is in.

## Scripts

The scripts to run are the `main_*.py` files in `pipeline/`, grouped into numbered steps in the order you would normally use them. Each can be run from the IDE (edit its `SETTINGS` block and press Run) or from the command line, e.g. `python pipeline/step4_evaluation/main_evaluate2d.py`, where flags override those settings (`--help` lists them). They work from any working directory and look for their config in the project root. Steps 0 to 2 each have their own config (`video_config.json`, `image_dataset_config.json` and `image_tiling_config.json`, with one section per script); steps 3 to 7 share `config.json`. Each writes to a timestamped `output/run_<datetime>/` folder, or into the run folder it works on.

**Step 0: video editing** (`pipeline/step0_video_editing/`)

- **`main_slice_videos.py`**: cuts the same time window (`start` to `end`) out of every video in a folder, using an ffmpeg stream copy (fast, no re-encoding; needs `ffmpeg` on the PATH). The folder and the window come from its own config, `video_config.json` (template: `config_examples/video_config.example.json`). The slices keep the videos' file names and go to `<videos_dir>/sliced` unless `output_dir` is set; existing slices are skipped.

**Step 1: image dataset creation** (`pipeline/step1_image_dataset_creation/`) - more tools to come

- **`main_reconstruct_cvat_dataset.py`**: rebuilds a YOLO dataset from a CVAT annotation export (zip) and the video it was annotated on. CVAT exports the labels of a video task without the images, so this pulls the labelled frames out of the video and writes `images/train/`, `labels/train/`, `train.txt` and `data.yaml`. Only frames with annotations are kept unless `--keep-empty`. A label's number is used as the frame index in the video, so the task must have been made from that video with frame step 1. Its settings are the `reconstruct_cvat` section of `image_dataset_config.json`.
- **`main_convert_dataset_format.py`**: converts a detection dataset between the YOLO and COCO formats. `yolo_to_coco` (FiftyOne) turns a YOLO dataset folder (`data.yaml`, `images/`, `labels/`) into `data/` plus `annotations/instances_<split>.json`; `coco_to_yolo` (Ultralytics) turns that layout back into `images/<split>/`, `labels/<split>/` and a `data.yaml` ready for training. The input is never changed, and `--symlink` links the images instead of copying them. Its settings are the `convert_format` section of `image_dataset_config.json`.

**Step 2: image tiling** (`pipeline/step2_image_tiling/`)

- **`main_tile_dataset.py`**: tiles a full-frame YOLO dataset into small square crops (`crop_size`), to train a detector that is then run with sliced (SAHI) inference. Each image gives tiles at random positions around its objects, so that every object ends up in a tile; a box cut by a tile's edge is kept only if enough of it is visible. A fraction of the images (`empty_fraction`, plus every image without labels) also gives an empty background tile, and these are spread evenly over a 3x3 grid of the frame. The output is a YOLO dataset with a `data.yaml`, ready for step 3. Afterwards it shows a few random tiles with their boxes drawn, to check the labels (`--no-visualise` to skip, `--visualise-only` to view tiles made earlier). `--fraction 0.1` tiles 10% of each split as a quick test. Its settings are the `tile` section of `image_tiling_config.json`.

**Step 3: YOLO training** (`pipeline/step3_training/`)

- **`main_train_detector.py`**: trains a YOLO detector with Ultralytics from the `training` section of the config: the dataset yaml, GPU, W&B project / group, YOLO version and size, the optional P2 head (an extra small-object detection layer, from `p2_model_yaml`) and the training arguments. With `load_pretrained`, the standard model's pretrained weights are loaded, also into the P2 architecture (backbone and P3-P5 heads; the P2 layers start random). Runs go to `output/training/<wandb_project>/<run name>/`, named like the W&B run, e.g. `yolo_s_12_base` (`_2`, `_3`, ... if the name is taken). To evaluate a trained model, set a model's `model_path` in `models` to the run name (e.g. `"model_path": "yolo_s_12_base"`) instead of a path; it resolves to that run's `weights/best.pt`.

**Step 4: evaluation** (`pipeline/step4_evaluation/`)

- **`main_evaluate2d.py`**: runs the enabled detectors on the dataset in the config and scores them against the ground-truth labels: mAP@0.2 / 0.5 / 0.5:0.95, P/R/F1, a side-by-side model comparison, plus optional track plots and annotated videos. Predictions are cached and reused while the model and frames are unchanged. With a batch config, each camera video becomes one input of the run, and that run is what the 3D scripts work on.

**Step 5: inference** (`pipeline/step5_inference/`)

- **`main_inference2d.py`**: runs the detectors without ground truth, on new recordings (`--inputs path/to/videos`). It writes a run folder with the same layout as an evaluation run, so the 3D scripts can use it directly.

**Step 6: 3D points** (`pipeline/step6_points3d/`)

- **`main_points3d.py`**: matches each frame's detections across the cameras (epipolar distance, Hungarian method) and triangulates them into 3D points, with no tracking. Useful for checking the calibration and the cross-camera matching.

**Step 7: 3D tracking** (`pipeline/step7_tracking3d/`)

- **`main_tracks3d.py`**: builds 3D tracks automatically. Triangulated points are linked into tracklets, each tracklet seeds a factor-graph track, and every 2D detection is associated with tracks iteratively, including detections seen by one camera only. Detections at fixed pixel positions (background clutter) are removed first. Results go to `<run>/3d/<model>/factor_graph/tracks_<datetime>/`: tracks (`tracks_3d.csv`), settings (`params.json`), 3D and reprojection plots, and, when the run has labels, `evaluation_3d.txt` / `evaluation_2d.txt`.
- **`main_tracks3d_guided.py`**: 3D tracks for hand-picked flies. An annotator opens each camera's video, where you click where each fly starts and / or ends. Later views can suggest the points from earlier views using the epipolar geometry. Each annotated fly gets exactly one track: the clicks fix its frame range and act as trusted constraints. Annotations are saved and can be reused with `--no-gui`. The annotator keeps the views in sync using each camera's `frame_offset`. Switching views opens the same instant, `g` jumps to the selected fly's point in another view, and if the same START / END is annotated in two views, the panel shows the offset that would line them up. To find an unknown offset, annotate one sharp event, like a take-off, in every view. Outputs and evaluation are the same as for `main_tracks3d.py`, in `factor_graph/guided_<datetime>/`.

Use `--source gt` on the 3D scripts to run them on the ground-truth labels instead of the predictions.

The 3D scripts also draw the red fake fruit, the green leaves around it and the arena frame, with no annotation needed. The fruit and leaves come from each view's median frame (frames sampled across the video, so the flies vanish), segmented in HSV:

- **Fruit:** round, saturated red blobs are candidates. The combination of one blob per view that triangulates consistently is the fruit, which gives the centre of its base on the wall. Its radius is measured from the blobs. It is drawn as a shaded hemisphere bulging out from the wall, towards the cameras.
- **Leaves:** the green region around the fruit in each view is laid flat onto the fruit's wall (camera rays meet the wall plane), and kept where at least two views agree. The terminal prints how well each view agrees (IoU), a quick check of the calibration.
- **Arena frame:** the tapered box of pipes is grown out from the fruit using its measured size (`arena_plotting.arena`): the back face on the fruit's wall, upright, at a fixed offset from the fruit, and the front face `depth` towards the cameras. It is only as accurate as the fruit's placement on the back wall, so check the dashed frame in the reprojection images.

The fruit and leaves are fitted once per run and cached in `<run>/3d/fruit/` and `<run>/3d/leaves/`, each with one debug image per camera (`hsv/<camera>.jpg`). The settings are in `arena_plotting` in the config. The frame needs no fitting. Use `--no-arena` to skip all three.

## Layout

- **`pipeline/`**: the `main_*.py` scripts, one folder per step (see Scripts). They only parse settings and call into the packages below.
- **`dataset_tools/`**: dataset creation, cleaning and conversion. `image_tools/tiling/tile_dataset.py` tiles a full-frame dataset into training crops; `image_tools/yolo_format.py` reads YOLO datasets from disk; `image_tools/cvat/reconstruct_dataset.py` rebuilds a dataset from a CVAT export and its video; `image_tools/conversion/` holds the YOLO -> COCO (`convert_yolo_to_coco.py`) and COCO -> YOLO (`convert_coco_to_yolo.py`) converters; `video_tools/slice_videos.py` cuts a time window out of a folder of videos.
- **`visualise/`**: graphics shared across the pipeline. `yolo_dataset.py` draws YOLO boxes on an image and shows random samples of a dataset (used by the tiling step to check its tiles).
- **`training/`**: building the YOLO model (version, size, P2 head, pretrained weights) and running Ultralytics training.
- **`evaluate2d/`**: datasets, detectors, metrics, prediction caching and 2D visualisation.
- **`tracking3d/`**: camera model and triangulation, run / camera loading, the factor graph (unguided and guided), evaluation, reports and 3D plots.
- **`gui/`**: the start / end annotator and its epipolar suggestions.
- **`arena_plotting/`**: static arena objects for the 3D plots. `fruit/` and `leaves/` each have `hsv/` (segmentation), `mesh/` (reconstruction and drawing) and `locate.py` (per-run glue and cache); `arena/` has `mesh/` and `locate.py` only, since nothing is segmented. `objects.py` bundles them for the plots, and `geometry.py`, `drawing.py`, `frames.py` and `settings.py` hold the code they share.
