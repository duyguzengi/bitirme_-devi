"""
Dataset package for Space Debris Edge AI Pipeline.
"""

from .synthetic_debris_generator import DebrisObject, SpaceDebrisSimulator
from .label_converter import (
    xyxy_to_yolo,
    yolo_to_xyxy,
    endpoints_to_yolo,
    export_yolo_annotation_file,
    export_mot_gt_file,
)
from .dataset_downloader import DATASET_REGISTRY, generate_local_benchmark_dataset

__all__ = [
    "DebrisObject",
    "SpaceDebrisSimulator",
    "xyxy_to_yolo",
    "yolo_to_xyxy",
    "endpoints_to_yolo",
    "export_yolo_annotation_file",
    "export_mot_gt_file",
    "DATASET_REGISTRY",
    "generate_local_benchmark_dataset",
]
