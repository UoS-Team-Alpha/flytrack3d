"""3D reconstruction from a batch evaluation run's 2D detections
(main_points3d.py, main_tracks3d.py).

    multiview     choose a run, pair cameras with its inputs, load detections per view
    camera        camera model, epipolar geometry, linear triangulation of one point
    triangulate   cross-view Hungarian matching, triangulation of every frame
    factor_graph  gtsam track estimation (reprojection + constant-velocity factors)
    evaluate      tracks vs ground truth in 3D, and 2D metrics with the tracks back-projected
    past_results  find and load saved 3D results, to plot them again
    external      a csv of 3D points / tracks or folders of YOLO labels from outside a run

Its 3D plots, reprojection overlays, back-projection videos and text reports are in visualise/.
"""
