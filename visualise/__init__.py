"""Everything that is drawn or laid out for people to look at: graphs, box overlays and
metric tables. The packages that compute the results (evaluate2d, tracking3d) call into here.

    plots2d/   2D graphs, the size of a camera frame
        track_plots   ground-truth and predicted 2D tracks of each sequence as lines
        reprojection  3D tracks projected back into each camera (the calibration check)
    plots3d/   3D graphs
        scene         triangulated points and factor-graph tracks, with ground-truth,
                      annotation and arena overlays (interactive or saved)
    boxes/     bounding boxes drawn on images and videos
        yolo_dataset  draw YOLO boxes on an image; show random samples of a YOLO dataset
        video         annotated videos: ground truth / detector boxes, and the factor graph's
                      back-projected boxes
    metrics/   metrics as text tables, for the terminal and the .txt reports
        detection     2D detector reports: per model, model comparison, per-input batch table
        tracking      3D track summary, 3D evaluation, 2D evaluation with back-projection
    common     colours and small helpers the above share

The arena objects drawn into the 3D plots (fruit, leaves, frame) are fitted and drawn by
arena_plotting.
"""
