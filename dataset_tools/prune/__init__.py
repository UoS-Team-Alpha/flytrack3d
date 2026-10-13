"""Dataset pruning: find and remove redundant frames from a YOLO dataset built from video.

A fly that sits still gets the same bounding box size in hundreds of consecutive frames. These
near-identical frames inflate the dataset without adding information. They are found, shown for
inspection, and left out of a pruned copy of the dataset. The original is never modified.

    redundancy  rank a video's bbox sizes and decide with SSIM which boxes are kept / redundant
    plan        which frames a run drops: whole frames only, with a safety margin and a cap
    runs        manifests (the frames of a run) and the purge log (the status of every run)
    build       the pruned dataset: the original minus every applied run, images hard-linked
    guided      the workflow: a whole split, one video prefix at a time, resumable
    review      choose the frames to look at: a rank's frames, a run's dropped / kept pairs
                (drawn by visualise.boxes.pruning)
    dataset     the dataset's label files, video prefixes and images
    metrics     frames and instances per split and video prefix, as a saved CSV
    prompts     the y/n and pick-a-number questions
    paths       where all of this is written (output/pruned_datasets/)
"""
