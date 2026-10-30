"""
Space Debris Kalman Filter Module.
===================================
Implements an 8-state Kalman Filter tailored for tracking tumbling space debris in 2D image coordinates:
    State: x = [center_x, center_y, aspect_ratio, height, vx, vy, va, vh]^T
    Measurement: z = [center_x, center_y, aspect_ratio, height]^T

Features:
- Constant Velocity (CV) kinematic motion model with discrete time step dt.
- Adaptive covariance estimation based on detection confidence and bounding box size.
- Trajectory smoothing and velocity extrapolation.
"""

import numpy as np


class SpaceKalmanFilter:
    """Kalman filter for tracking bounding boxes in image space."""

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        self.ndim = 4
        self.dt = dt

        # State Transition Matrix F (8x8)
        self.F = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = dt

        # Measurement Matrix H (4x8)
        self.H = np.zeros((self.ndim, 2 * self.ndim), dtype=np.float32)
        for i in range(self.ndim):
            self.H[i, i] = 1.0

        # Motion noise standard deviations
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray):
        """
        Creates track state from initial unassociated measurement.
        measurement: [cx, cy, aspect_ratio, height]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2.0 * self._std_weight_position * measurement[3],
            2.0 * self._std_weight_position * measurement[3],
            1e-2,
            2.0 * self._std_weight_position * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            1e-5,
            10.0 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean.astype(np.float32), covariance.astype(np.float32)

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Predicts the next state given the current state and covariance."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self.F, mean)
        covariance = np.linalg.multi_dot((self.F, covariance, self.F.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Projects the state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self.H, mean)
        covariance = np.linalg.multi_dot((self.H, covariance, self.H.T)) + innovation_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Runs the Kalman correction step with a new measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy_cholesky(projected_cov)
        kalman_gain = scipy_cho_solve((chol_factor, lower), np.dot(covariance, self.H.T).T).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def scipy_cholesky(matrix):
    """Robust Cholesky decomposition with positive-definite fallback."""
    try:
        return np.linalg.cholesky(matrix), True
    except np.linalg.LinAlgError:
        # Add small jitter to diagonal if non-PD
        matrix_jittered = matrix + np.eye(matrix.shape[0]) * 1e-4
        return np.linalg.cholesky(matrix_jittered), True


def scipy_cho_solve(chol_tuple, b):
    """Solves A x = b using Cholesky factor L."""
    L, _ = chol_tuple
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x
