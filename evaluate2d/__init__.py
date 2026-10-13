"""2D detector evaluation (main_evaluate2d.py).

    detectors        SAHI / Ultralytics detector wrappers
    metrics          COCO-style mAP and precision / recall
    predictions      saved predictions (YOLO labels per frame) and their cache key
    evaluate_models  the evaluation pipeline: cached inference, scoring, reports

The datasets it evaluates are loaded by common/datasets.py. Its track plots, annotated videos and text reports are in visualise/.
"""
