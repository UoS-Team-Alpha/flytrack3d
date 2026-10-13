"""Dataset creation, cleaning and conversion.

    stats                 per-split statistics of a YOLO dataset: videos, images, instances and box sizes (main_analyse_dataset.py)
    convert_yolo_to_coco  YOLO dataset (data.yaml + images / labels) -> COCO json, with FiftyOne (main_convert_dataset_format.py)
    convert_coco_to_yolo  COCO json -> YOLO labels, images and data.yaml, with Ultralytics (main_convert_dataset_format.py)
    cvat                  CVAT label export (zip) + the annotated video -> YOLO dataset of the labelled frames (main_reconstruct_cvat_dataset.py)
    merge                 a folder of YOLO datasets -> one YOLO dataset, with no image in it twice (main_merge_datasets.py)
    tile                  crop tiles around the objects of a YOLO dataset, plus stratified empty background tiles (main_tile_dataset.py)
    slice_videos          cut the same time window out of every video in a folder, with ffmpeg (main_slice_videos.py)
    prune/                find and remove redundant frames from a YOLO dataset built from video (main_analyse_redundancy.py, main_guided_prune.py)
"""
