# Step 1: image dataset creation

Part of the [pipeline](../README.md), which explains how to run the scripts. More tools to come.

All settings are in `configs/image_dataset.json`, one section per script.

| Script | Config section |
| --- | --- |
| [`main_reconstruct_cvat_dataset.py`](#main_reconstruct_cvat_datasetpy) | `reconstruct_cvat` |
| [`main_merge_datasets.py`](#main_merge_datasetspy) | `merge` |
| [`main_convert_dataset_format.py`](#main_convert_dataset_formatpy) | `convert_format` |
| [`main_analyse_dataset.py`](#main_analyse_datasetpy) | `analyse` |
| [`main_check_bboxes.py`](#main_check_bboxespy) | `check_bboxes` |

## `main_reconstruct_cvat_dataset.py`

Rebuilds a YOLO dataset from a CVAT annotation export (zip) and the video it was annotated on. CVAT exports the labels of a video task without the images, so this pulls the labelled frames out of the video.

- **Output:** `images/train/`, `labels/train/`, `train.txt` and `data.yaml`.
- **Empty frames:** only frames with annotations are kept unless `--keep-empty`.
- **Requirement:** a label's number is used as the frame index in the video, so the task must have been made from that video with frame step 1.

## `main_merge_datasets.py`

Merges a folder of YOLO datasets (one per sub-folder) into one dataset.

- **Output:** `images/<split>/<dataset>_<image>`, `labels/<split>/`, `<split>.txt` and `data.yaml`.
- **Duplicates:** the same image is never added twice. The images already in the output (it may hold an earlier merge, which is then added to) and every image to add are hashed over the whole file, and an image whose hash is already there, in any split, is skipped. The skipped duplicates are reported, as are those whose labels differ from the copy that was kept.
- **Flags:**
  - `--keep-classes` keeps only those classes' boxes (renumbered from 0).
  - `--no-keep-empty` drops images left without a box.
- **Safety:** nothing in the output is overwritten and the merged datasets are never changed.

## `main_convert_dataset_format.py`

Converts a detection dataset between the YOLO and COCO formats.

- **`yolo_to_coco`** (FiftyOne): turns a YOLO dataset folder (`data.yaml`, `images/`, `labels/`) into `data/` plus `annotations/instances_<split>.json`.
- **`coco_to_yolo`** (Ultralytics): turns that layout back into `images/<split>/`, `labels/<split>/` and a `data.yaml` ready for training.

The input is never changed, and `--symlink` links the images instead of copying them.

## `main_analyse_dataset.py`

Reports what a YOLO dataset holds, without changing it.

- **Report:** for each split (train / val / test) it lists the video prefixes with their image, background-image and instance counts, the image sizes, and the distribution of the box sizes and of the instances per image.
- **Warnings:** when a video is in more than one split, when images and label files do not pair up, and when boxes reach outside their image.
- **Figures:**
  - pie charts of the images and instances per split and per video;
  - per split, heatmaps of where the boxes sit and of their width against their height, with a histogram of their size.
- **Video names:** a frame's video is the part of its file name before `_frame_`, else its sub-folder of the split.
- **Output:** the report and figures are saved to `output/dataset_analysis/<dataset name>/`.
- **Flags:**
  - `--no-save` skips saving.
  - `--no-show` opens no windows.
  - `--samples` also opens a few random images with their boxes drawn.

## `main_check_bboxes.py`

Opens a few random images of a YOLO dataset with their boxes and class names drawn, one at a time (press any key for the next), to check the labels line up. Nothing is saved and the dataset is never changed.

- **Flags:**
  - `-n` sets how many images.
  - `--split` sets which split they come from.
  - `--seed` shows the same ones every run.
