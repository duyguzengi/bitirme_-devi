"""
Synthetic Space Debris and Starfield Generator.

Directly implements the observation-based simulation pipeline described in
the SDTD paper (Zhuang, Song et al., 2025, Section 3.2):
- Line source parameterization: S_t = {x_t, y_t, l, w, theta, v} (Eq. 4)
- Trajectory propagation: x_2 = x_1 + v * cos(theta) * dt (Eq. 5)
- Point Spread Function (PSF) Gaussian diffusion: (Eq. 6)
- Synthetic starfield background + sky brightness noise
"""

import math
import random
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class DebrisObject:
    """Represents a simulated space debris instance with physical trajectory."""

    def __init__(
        self,
        debris_id: int,
        x: float,
        y: float,
        length: float,
        width: float,
        angle_rad: float,
        velocity: float,
        intensity: float = 230.0,
    ):
        self.debris_id = debris_id
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.angle_rad = angle_rad
        self.velocity = velocity
        self.intensity = intensity

    def update(self, dt: float = 1.0):
        """Propagate position based on velocity vector (Equation 5)."""
        self.x += self.velocity * math.cos(self.angle_rad) * dt
        self.y += self.velocity * math.sin(self.angle_rad) * dt

    def get_endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Computes two endpoints of the linear streak (SDTD format)."""
        half_l = self.length / 2.0
        dx = half_l * math.cos(self.angle_rad)
        dy = half_l * math.sin(self.angle_rad)
        st = (self.x - dx, self.y - dy)
        ed = (self.x + dx, self.y + dy)
        return st, ed

    def get_bbox(self, img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns bounding box [x_min, y_min, x_max, y_max].
        Returns None if object is completely outside image bounds.
        """
        st, ed = self.get_endpoints()
        margin = max(self.width * 2.0, 4.0)

        x_min = max(0.0, min(st[0], ed[0]) - margin)
        y_min = max(0.0, min(st[1], ed[1]) - margin)
        x_max = min(float(img_w), max(st[0], ed[0]) + margin)
        y_max = min(float(img_h), max(st[1], ed[1]) + margin)

        if x_max <= x_min or y_max <= y_min:
            return None
        return (x_min, y_min, x_max, y_max)


class SpaceDebrisSimulator:
    """Generates synthetic multi-frame video sequences with space debris and starfield."""

    def __init__(
        self,
        width: int = 640,
        height: int = 640,
        num_stars: int = 80,
        psf_sigma: float = 1.2,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.num_stars = num_stars
        self.psf_sigma = psf_sigma
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.starfield = self._generate_static_starfield()

    def _generate_static_starfield(self) -> np.ndarray:
        """Creates fixed celestial star field with Poisson/Gaussian noise."""
        bg = np.zeros((self.height, self.width), dtype=np.float32)

        # Baseline skylight / cosmic glow
        sky_noise = np.random.normal(loc=12.0, scale=3.5, size=(self.height, self.width))
        bg = np.clip(bg + sky_noise, 0, 255)

        # Random point stars with varying brightness
        for _ in range(self.num_stars):
            sx = random.randint(5, self.width - 6)
            sy = random.randint(5, self.height - 6)
            brightness = random.uniform(80.0, 255.0)
            radius = random.choice([1, 1, 1, 2])
            cv2.circle(bg, (sx, sy), radius, float(brightness), -1)

        # Apply slight blur to simulate optical PSF for stars
        bg = cv2.GaussianBlur(bg, (3, 3), 0.8)
        return bg

    def generate_sequence(
        self,
        num_frames: int = 15,
        num_debris: int = 2,
    ) -> Tuple[List[np.ndarray], List[List[Dict]]]:
        """
        Generates a sequence of frames and corresponding ground-truth annotations.

        Returns:
            frames: List of uint8 images.
            annotations: List of annotation dicts for each frame.
        """
        debris_list: List[DebrisObject] = []
        for i in range(num_debris):
            d = DebrisObject(
                debris_id=i + 1,
                x=random.uniform(50, self.width - 50),
                y=random.uniform(50, self.height - 50),
                length=random.uniform(12.0, 35.0),
                width=random.uniform(1.5, 3.0),
                angle_rad=random.uniform(0, 2 * math.pi),
                velocity=random.uniform(3.0, 8.0),
                intensity=random.uniform(170.0, 255.0),
            )
            debris_list.append(d)

        frames = []
        all_annos = []

        for f_idx in range(num_frames):
            frame_bg = self.starfield.copy()
            frame_annos = []

            debris_canvas = np.zeros((self.height, self.width), dtype=np.float32)

            for d in debris_list:
                bbox = d.get_bbox(self.width, self.height)
                if bbox is not None:
                    st, ed = d.get_endpoints()
                    # Draw linear streak (SDTD line source)
                    cv2.line(
                        debris_canvas,
                        (int(st[0]), int(st[1])),
                        (int(ed[0]), int(ed[1])),
                        float(d.intensity),
                        max(1, int(round(d.width))),
                        cv2.LINE_AA,
                    )

                    frame_annos.append({
                        "frame_id": f_idx,
                        "debris_id": d.debris_id,
                        "class_id": 0,
                        "class_name": "space_debris",
                        "bbox_xyxy": bbox,
                        "endpoints": (st, ed),
                        "center": (d.x, d.y),
                        "velocity": d.velocity,
                    })

                # Propagate for next frame
                d.update(dt=1.0)

            # Apply PSF Gaussian diffusion (Eq. 6 from paper)
            ksize = int(math.ceil(self.psf_sigma * 3)) * 2 + 1
            diffused_debris = cv2.GaussianBlur(debris_canvas, (ksize, ksize), self.psf_sigma)

            # Combine background starfield with debris
            combined = np.clip(frame_bg + diffused_debris, 0.0, 255.0).astype(np.uint8)
            frames.append(combined)
            all_annos.append(frame_annos)

        return frames, all_annos
