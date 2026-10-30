"""
Visualization and Diagnostic Utility Module.

Provides functions to visualize:
1. Star removal and CLAHE before/after comparisons.
2. Ground-truth bounding boxes and debris trajectory streaks.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


def draw_annotations(
    image: np.ndarray,
    annotations: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    draw_endpoints: bool = True,
) -> np.ndarray:
    """Draws bounding boxes and labels onto the image."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    for anno in annotations:
        bbox = anno.get("bbox_xyxy")
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

            # Draw label
            tid = anno.get("debris_id", 0)
            label = f"Debris #{tid}"
            cv2.putText(
                canvas,
                label,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        if draw_endpoints and "endpoints" in anno:
            st, ed = anno["endpoints"]
            cv2.circle(canvas, (int(st[0]), int(st[1])), 2, (0, 0, 255), -1)
            cv2.circle(canvas, (int(ed[0]), int(ed[1])), 2, (255, 0, 0), -1)

    return canvas


def create_comparison_grid(
    raw_img: np.ndarray,
    cosmic_cleaned: np.ndarray,
    star_removed: np.ndarray,
    enhanced: np.ndarray,
) -> np.ndarray:
    """Combines 4 pipeline stages into a 2x2 comparison grid for reporting and debugging."""
    def to_bgr(im):
        im_u = np.clip(im, 0, 255).astype(np.uint8)
        return cv2.cvtColor(im_u, cv2.COLOR_GRAY2BGR) if len(im.shape) == 2 else im_u

    b_raw = to_bgr(raw_img)
    b_cosmic = to_bgr(cosmic_cleaned)
    b_stars = to_bgr(star_removed)
    b_enh = to_bgr(enhanced)

    # Put title texts
    cv2.putText(b_raw, "1. Raw Space Frame", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(b_cosmic, "2. Hot Pixel Cleaned", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(b_stars, "3. Star Field Removed", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(b_enh, "4. ZScale & CLAHE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    top_row = np.hstack([b_raw, b_cosmic])
    bottom_row = np.hstack([b_stars, b_enh])
    grid = np.vstack([top_row, bottom_row])
    return grid
