"""3D reconstruction from a batch evaluation run's 2D detections
(main_points3d.py, main_tracks3d.py).

    multiview     choose a run, pair cameras with its inputs, load detections per view
    triangulate   camera model, cross-view Hungarian matching, triangulation
    factor_graph  gtsam track estimation (reprojection + constant-velocity factors)
    visualise     3D plots and per-camera reprojection overlays
    evaluate      tracks vs ground truth in 3D, and 2D metrics with the tracks back-projected
"""
