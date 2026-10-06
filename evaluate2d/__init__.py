"""2D detector evaluation (main_evaluate2d.py).

    config           config loading shared by every main_*.py
    datasets         ground-truth datasets: image folders, videos, batches of both
    detectors        SAHI / Ultralytics detector wrappers
    metrics          COCO-style mAP and precision / recall
    evaluate_models  the evaluation pipeline: cached inference, scoring, reports
    visualise        annotated videos and 2D track plots
"""
