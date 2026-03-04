"""
Basketball-tuned Kalman Filter for ByteTrack.

Subclasses the boxmot KalmanFilterXYAH to adjust process and measurement
noise parameters for basketball player dynamics.

Basketball motion characteristics:
    - Player speed: 3-8 m/s (fast cuts, sprints, deceleration)
    - High jerk: sudden direction changes (screens, drives, defensive slides)
    - Typical bounding box height: 100-300 pixels at 720p
    - Frame rate: 25 FPS (SportsMOT)

Tuning rationale:
    - _std_weight_position (1/15 vs default 1/20):
        Increased position uncertainty allows the filter to accept larger
        deviations between predicted and observed positions. This prevents
        ID switches during fast crossovers where the Kalman prediction
        lags behind the actual player position.

    - _std_weight_velocity (1/100 vs default 1/160):
        Increased velocity process noise models the high acceleration
        variability in basketball. Players frequently go from stationary
        to sprinting (or vice versa) within 2-3 frames. The higher noise
        allows the velocity estimate to adapt more quickly.

    The noise is proportional to bounding box height, which acts as a
    proxy for distance from camera — farther players have smaller boxes
    and proportionally smaller pixel-space motion.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Basketball-tuned constants
BASKETBALL_STD_WEIGHT_POSITION = 1.0 / 15  # default: 1/20
BASKETBALL_STD_WEIGHT_VELOCITY = 1.0 / 100  # default: 1/160


class BasketballKalmanFilter:
    """
    Kalman Filter tuned for basketball player tracking.

    State vector: [x, y, a, h, vx, vy, va, vh]
        x, y   — bounding box center position
        a      — aspect ratio (width / height)
        h      — bounding box height
        vx, vy — velocity of center
        va     — rate of change of aspect ratio
        vh     — rate of change of height

    The filter uses a constant-velocity motion model with basketball-specific
    noise parameters. Measurement noise is scaled by detection confidence
    (NSA Kalman approach) when confidence is available.

    Attributes:
        _std_weight_position: Position noise scaling factor.
        _std_weight_velocity: Velocity noise scaling factor.
        _motion_mat: State transition matrix (8x8).
        _update_mat: Measurement matrix (4x8).
    """

    ndim = 4  # Measurement dimensions (x, y, a, h)
    dt = 1.0  # Time step (1 frame)

    def __init__(
        self,
        std_weight_position: float = BASKETBALL_STD_WEIGHT_POSITION,
        std_weight_velocity: float = BASKETBALL_STD_WEIGHT_VELOCITY,
    ):
        """
        Initialize the Kalman Filter with basketball-tuned noise parameters.

        Args:
            std_weight_position: Scaling factor for position uncertainty.
                Higher values = more tolerant of position deviations.
            std_weight_velocity: Scaling factor for velocity uncertainty.
                Higher values = faster adaptation to acceleration changes.
        """
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # State transition matrix: constant velocity model
        # [x, y, a, h, vx, vy, va, vh] → [x+vx, y+vy, a+va, h+vh, vx, vy, va, vh]
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt

        # Measurement matrix: observe [x, y, a, h] from state
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        logger.debug(
            f"BasketballKF initialized: pos_weight={std_weight_position:.4f}, "
            f"vel_weight={std_weight_velocity:.6f}"
        )

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from an unassociated measurement.

        Args:
            measurement: Bounding box (x, y, a, h) — center position,
                aspect ratio, and height.

        Returns:
            Tuple of (mean, covariance):
                mean: Initial state vector (8,).
                covariance: Initial state covariance matrix (8, 8).
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initial covariance — large for velocity (unknown)
        std = [
            2 * self._std_weight_position * measurement[3],   # x
            2 * self._std_weight_position * measurement[3],   # y
            1e-2,                                               # a
            2 * self._std_weight_position * measurement[3],   # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,                                               # va
            10 * self._std_weight_velocity * measurement[3],  # vh
        ]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state (run the Kalman Filter prediction step).

        Args:
            mean: Current state mean (8,).
            covariance: Current state covariance (8, 8).

        Returns:
            Tuple of (predicted_mean, predicted_covariance).
        """
        # Process noise: proportional to bounding box height
        std_pos = [
            self._std_weight_position * mean[3],    # x
            self._std_weight_position * mean[3],    # y
            1e-2,                                    # a
            self._std_weight_position * mean[3],    # h
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],    # vx
            self._std_weight_velocity * mean[3],    # vy
            1e-5,                                    # va
            self._std_weight_velocity * mean[3],    # vh
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = (
            self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        )

        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray, confidence: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project the state to measurement space.

        Uses NSA Kalman (confidence-scaled measurement noise):
        higher confidence → lower measurement noise → more trust in detection.

        Args:
            mean: State mean (8,).
            covariance: State covariance (8, 8).
            confidence: Detection confidence [0, 1]. Higher = less noise.

        Returns:
            Tuple of (projected_mean, projected_covariance) in measurement space.
        """
        std = [
            self._std_weight_position * mean[3],  # x
            self._std_weight_position * mean[3],  # y
            1e-1,                                   # a
            self._std_weight_position * mean[3],  # h
        ]
        innovation_cov = np.diag(np.square(std))

        # NSA Kalman: scale noise inversely with confidence
        if confidence > 0:
            innovation_cov *= (1 - confidence)

        mean_proj = self._update_mat @ mean
        covariance_proj = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )

        return mean_proj, covariance_proj

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the Kalman Filter correction step.

        Args:
            mean: Predicted state mean (8,).
            covariance: Predicted state covariance (8, 8).
            measurement: Observed bounding box (x, y, a, h).
            confidence: Detection confidence for NSA scaling.

        Returns:
            Tuple of (corrected_mean, corrected_covariance).
        """
        projected_mean, projected_cov = self.project(
            mean, covariance, confidence
        )

        # Kalman gain
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol_factor,
            np.linalg.solve(
                chol_factor, (covariance @ self._update_mat.T).T
            ),
        ).T

        # Innovation (residual)
        innovation = measurement - projected_mean

        # Update
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """
        Compute gating distance (Mahalanobis) between state and measurements.

        Used for gating associations — reject matches that are too far.

        Args:
            mean: State mean (8,).
            covariance: State covariance (8, 8).
            measurements: Array of measurements (N, 4).
            only_position: If True, use only position (x, y) for gating.

        Returns:
            Array of squared Mahalanobis distances (N,).
        """
        mean_proj, cov_proj = self.project(mean, covariance)

        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj = cov_proj[:2, :2]
            measurements = measurements[:, :2]

        diff = measurements - mean_proj
        cholesky = np.linalg.cholesky(cov_proj)
        z = np.linalg.solve(cholesky, diff.T).T

        return np.sum(z * z, axis=1)
