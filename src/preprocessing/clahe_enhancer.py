"""
Astronomical Contrast Enhancement and CLAHE Module.

Implements the ZScale contrast stretch algorithm (from SDTD paper Zhuang et al., 2025)
and Contrast Limited Adaptive Histogram Equalization (CLAHE) tailored for
low-contrast deep-space backgrounds and faint space debris streaks.
"""

import cv2
import numpy as np


def apply_zscale(image: np.ndarray, k: float = 2.5) -> np.ndarray:
    """
    Applies the ZScale algorithm (Equations 1-3 from SDTD paper).

    z1 = z_median - k * z_std
    z2 = z_median + k * z_std
    I_scaled = 255 * (I - z1) / (z2 - z1) clipped to [0, 255]

    Args:
        image: Input grayscale image (float or uint8).
        k: Contrast scale multiplier (default 2.5 from paper).

    Returns:
        Enhanced uint8 image in range [0, 255].
    """
    img_f = image.astype(np.float32)
    z_median = np.median(img_f)
    z_std = np.std(img_f)

    z1 = z_median - k * z_std
    z2 = z_median + k * z_std

    if abs(z2 - z1) < 1e-5:
        return np.clip(img_f, 0, 255).astype(np.uint8)

    scaled = 255.0 * (img_f - z1) / (z2 - z1)
    scaled = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return scaled


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Prevents over-amplification of background noise while highlighting faint line tracks.

    Args:
        image: Grayscale uint8 image.
        clip_limit: Threshold for contrast limiting (1.5 - 3.0 typical).
        tile_grid_size: Grid size for local histogram equalization.

    Returns:
        Contrast-enhanced uint8 image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced


def enhance_space_image(image: np.ndarray, use_zscale: bool = True, clip_limit: float = 2.0) -> np.ndarray:
    """
    Complete astronomical contrast enhancement pipeline.
    Combines ZScale stretching with CLAHE.
    """
    if use_zscale:
        stretched = apply_zscale(image)
    else:
        stretched = image.copy()

    enhanced = apply_clahe(stretched, clip_limit=clip_limit)
    return enhanced
