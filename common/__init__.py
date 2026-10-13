"""What every other package builds on. It imports none of them.

    config       config loading shared by every main_*.py
    datasets     frames and datasets: image folders, videos, batches of both
    yolo_format  read a YOLO dataset from disk: split folders, images, label files
    linking      link per-frame points into tracks (no track IDs in labels / detections)
"""
