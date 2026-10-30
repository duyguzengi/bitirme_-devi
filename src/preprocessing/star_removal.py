"""
Star Removal and Background Subtraction Module.

In astronomical and on-orbit star tracker imaging:
- Fixed celestial stars appear as point sources.
- Satellite roll/attitude creates uniform global celestial motion.
- Space debris has independent high relative angular velocity (linear streak / fast shift).

This module provides:
1. Morphological Top-Hat / Opening Filtering (single-frame star suppression).
2. Temporal Frame Differencing (two-frame motion isolation).
3. Running Background Subtraction (multi-frame temporal median filter).
"""

import cv2
import numpy as np
from typing import List, Optional


def remove_stars_morphological(image: np.ndarray, star_kernel_size: int = 3, streak_kernel_len: int = 9) -> np.ndarray:
    """
    Suppresses point-like star spots while preserving elongated linear debris streaks.

    Uses directional opening filters to separate compact isotropic stars from
    anisotropic line sources.

    Args:
        image: Grayscale uint8 image.
        star_kernel_size: Approximate diameter of point stars in pixels.
        streak_kernel_len: Expected minimum length of debris streak.

    Returns:
        Debris-enhanced image with stars attenuated.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Directional kernels for streak preservation (0, 45, 90, 135 degrees)
    k_len = streak_kernel_len
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1)),  # Horizontal
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len)),  # Vertical
        np.eye(k_len, dtype=np.uint8),                          # 45 deg diagonal
        np.fliplr(np.eye(k_len, dtype=np.uint8)),               # 135 deg diagonal
    ]

    # Morphological opening with line kernels preserves only elongated features
    max_streak = np.zeros_like(gray)
    for k in kernels:
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
        max_streak = np.maximum(max_streak, opened)

    # Isotropic circular star kernel
    star_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (star_kernel_size, star_kernel_size))
    point_stars = cv2.morphologyEx(gray, cv2.MORPH_OPEN, star_kernel)

    # Subtract point stars and combine with directional streak mask
    star_attenuated = cv2.subtract(gray, point_stars)
    result = cv2.addWeighted(star_attenuated, 0.7, max_streak, 0.8, 0)

    return result


def frame_difference_stars(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: int = 15) -> np.ndarray:
    """
    Temporal differencing between consecutive frames.
    Static stars and smooth celestial background cancel out; moving debris creates a distinct signature.

    Args:
        prev_frame: Previous grayscale frame (uint8).
        curr_frame: Current grayscale frame (uint8).
        threshold: Absolute difference threshold.

    Returns:
        Difference image highlighting moving debris.
    """
    p_gray = prev_frame if len(prev_frame.shape) == 2 else cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    c_gray = curr_frame if len(curr_frame.shape) == 2 else cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(c_gray, p_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_TOZERO)
    return mask


class TemporalBackgroundSubtractor:
    """
    Maintains a rolling temporal buffer of frames to compute background median.
    Stars and sky glow remain in background; fast-moving debris is isolated as foreground.
    """

    def __init__(self, history_size: int = 5, threshold: int = 20):
        self.history_size = history_size
        self.threshold = threshold
        self.buffer: List[np.ndarray] = []

    def update(self, frame: np.ndarray) -> np.ndarray:
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.buffer.append(gray)
        if len(self.buffer) > self.history_size:
            self.buffer.pop(0)

        if len(self.buffer) < 3:
            return gray

        # Temporal median represents static star field & sky background
        stack = np.stack(self.buffer, axis=0)
        background = np.median(stack, axis=0).astype(np.uint8)

        # Foreground is the positive difference (debris passing by)
        foreground = cv2.subtract(gray, background)
        _, thresh = cv2.threshold(foreground, self.threshold, 255, cv2.THRESH_TOZERO)
        return thresh
