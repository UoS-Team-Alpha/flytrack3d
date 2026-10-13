"""Everything that is drawn or laid out for people to look at: graphs, box overlays and
metric tables. The packages that compute the results (evaluate2d, tracking3d, dataset_tools)
are called first, and what they return is passed in here.

    plots2d/   2D graphs, the size of a camera frame
        track_plots   ground-truth and predicted 2D tracks of each sequence as lines
        reprojection  3D tracks projected back into each camera (the calibration check)
    plots3d/   3D graphs
        scene         triangulated points and factor-graph tracks, with ground-truth,
                      annotation and arena overlays (interactive or saved)
    charts/    statistical charts: pies, histograms and heatmaps
        dataset       what a YOLO dataset holds (pies per split and video), box positions and sizes
    boxes/     bounding boxes drawn on images and videos
        yolo_dataset  draw YOLO boxes on an image; show random samples of a YOLO dataset
        pruning       dataset pruning checks: frames of a bbox size rank, dropped frames next
                      to the kept frame they matched
        video         annotated videos: ground truth / detector boxes, and the factor graph's
                      back-projected boxes
    metrics/   metrics as text tables, for the terminal and the .txt reports
        detection     2D detector reports: per model, model comparison, per-input batch table
        tracking      3D track summary, 3D evaluation, 2D evaluation with back-projection
        dataset       dataset analysis: splits, videos, counts and box sizes
    common     colours and small helpers the above share

The arena objects drawn into the 3D plots (fruit, leaves, frame) are fitted and drawn by
arena_plotting.
"""
