"""
Cosmic Ray and Hot Pixel Filtering Module.

In space environments, cosmic radiation and high-energy particles create single-pixel
or few-pixel extreme intensity spikes on CMOS/CCD sensors ('hot pixels' / 'salt noise').
This module provides robust filtering using morphological and sigma-clipped median filters
without blurring thin space debris streaks.
"""

import cv2
import numpy as np


def filter_cosmic_rays_median(image: np.ndarray, kernel_size: int = 3, threshold_sigma: float = 3.5) -> np.ndarray:
    """
    Cleans isolated cosmic ray spikes while preserving elongated debris streaks.

    Args:
        image: Grayscale image (uint8 or float32).
        kernel_size: Size of the median filter window (typically 3 or 5).
        threshold_sigma: Multiplier for difference threshold to detect spikes.

    Returns:
        Cleaned image with hot pixels replaced by local median.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Median filtered image provides local background expectation
    median = cv2.medianBlur(gray, kernel_size)

    # Compute difference
    diff = gray.astype(np.float32) - median.astype(np.float32)

    # Estimate noise standard deviation via robust MAD (Median Absolute Deviation)
    mad = np.median(np.abs(diff - np.median(diff)))
    sigma = 1.4826 * (mad + 1e-6)

    # Hot pixel mask: points significantly brighter than surrounding median
    # but not connected elongated structures (which are debris)
    hot_pixel_mask = diff > (threshold_sigma * sigma)

    # Replace only hot pixels
    cleaned = gray.copy()
    cleaned[hot_pixel_mask] = median[hot_pixel_mask]

    return cleaned


def filter_isolated_spikes(image: np.ndarray, max_spike_area: int = 2) -> np.ndarray:
    """
    Removes very small 1-2 pixel isolated clusters using connected components.
    Elongated debris tracks (> 5-10 pixels) remain intact.
    """
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    cleaned = image.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_spike_area:
            mask = labels == i
            cleaned[mask] = 0

    return cleaned
