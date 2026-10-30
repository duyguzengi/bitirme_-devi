"""
Label Standardization and Conversion Module.

Converts various space debris annotation formats (SDTD line endpoints,
SPEED+ bounding boxes, COCO / Pascal VOC) to:
1. Standard YOLO Format: <class_id> <x_center> <y_center> <width> <height> (normalized to [0, 1])
2. MOTChallenge Format: <frame_id>,<track_id>,<x_min>,<y_min>,<width>,<height>,<conf>,-1,-1,-1
"""

from typing import Tuple, List, Dict
import os


def xyxy_to_yolo(
    bbox_xyxy: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    class_id: int = 0,
) -> str:
    """
    Converts absolute [x_min, y_min, x_max, y_max] to normalized YOLO line string:
    class_id x_center y_center width height
    """
    x_min, y_min, x_max, y_max = bbox_xyxy

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    xc = x_min + w / 2.0
    yc = y_min + h / 2.0

    # Normalize to [0, 1]
    norm_xc = max(0.0, min(1.0, xc / float(img_width)))
    norm_yc = max(0.0, min(1.0, yc / float(img_height)))
    norm_w = max(0.0, min(1.0, w / float(img_width)))
    norm_h = max(0.0, min(1.0, h / float(img_height)))

    return f"{class_id} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}"


def yolo_to_xyxy(
    yolo_line: str,
    img_width: int,
    img_height: int,
) -> Tuple[int, float, float, float, float]:
    """
    Converts normalized YOLO line string back to absolute [class_id, x_min, y_min, x_max, y_max].
    """
    parts = yolo_line.strip().split()
    class_id = int(parts[0])
    norm_xc, norm_yc, norm_w, norm_h = map(float, parts[1:5])

    w = norm_w * img_width
    h = norm_h * img_height
    xc = norm_xc * img_width
    yc = norm_yc * img_height

    x_min = xc - w / 2.0
    y_min = yc - h / 2.0
    x_max = xc + w / 2.0
    y_max = yc + h / 2.0

    return class_id, x_min, y_min, x_max, y_max


def endpoints_to_yolo(
    st: Tuple[float, float],
    ed: Tuple[float, float],
    margin_px: float,
    img_width: int,
    img_height: int,
    class_id: int = 0,
) -> str:
    """Converts SDTD line endpoints to YOLO bounding box format."""
    x_min = max(0.0, min(st[0], ed[0]) - margin_px)
    y_min = max(0.0, min(st[1], ed[1]) - margin_px)
    x_max = min(float(img_width), max(st[0], ed[0]) + margin_px)
    y_max = min(float(img_height), max(st[1], ed[1]) + margin_px)

    return xyxy_to_yolo((x_min, y_min, x_max, y_max), img_width, img_height, class_id)


def export_yolo_annotation_file(
    output_path: str,
    annotations: List[Dict],
    img_width: int,
    img_height: int,
):
    """Writes list of annotations for a single image to a .txt YOLO label file."""
    lines = []
    for anno in annotations:
        bbox = anno.get("bbox_xyxy")
        if bbox is not None:
            cid = anno.get("class_id", 0)
            lines.append(xyxy_to_yolo(bbox, img_width, img_height, cid))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def export_mot_gt_file(
    output_path: str,
    all_sequence_annos: List[List[Dict]],
):
    """
    Exports full multi-frame sequence to MOT format (gt.txt):
    <frame_idx>,<target_id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
    """
    lines = []
    for frame_idx, frame_annos in enumerate(all_sequence_annos):
        mot_frame = frame_idx + 1  # MOT frames are 1-indexed
        for anno in frame_annos:
            tid = anno.get("debris_id", 1)
            bbox = anno.get("bbox_xyxy")
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                w = x_max - x_min
                h = y_max - y_min
                lines.append(f"{mot_frame},{tid},{x_min:.2f},{y_min:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
