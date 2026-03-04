"""
Tests for the basketball-tuned Kalman Filter.

Validates the custom Kalman Filter parameters and predict/update cycle
for basketball player tracking scenarios.
"""

import numpy as np
import pytest

from src.tracking.kalman_filter import (
    BASKETBALL_STD_WEIGHT_POSITION,
    BASKETBALL_STD_WEIGHT_VELOCITY,
    BasketballKalmanFilter,
)


class TestBasketballKalmanFilterInit:
    """Tests for Kalman Filter initialization."""

    def test_default_parameters(self):
        """Test that default parameters match basketball-tuned values."""
        kf = BasketballKalmanFilter()

        assert kf._std_weight_position == BASKETBALL_STD_WEIGHT_POSITION
        assert kf._std_weight_velocity == BASKETBALL_STD_WEIGHT_VELOCITY
        assert kf._std_weight_position == pytest.approx(1 / 15, rel=1e-5)
        assert kf._std_weight_velocity == pytest.approx(1 / 100, rel=1e-5)

    def test_custom_parameters(self):
        """Test that custom parameters can be passed."""
        kf = BasketballKalmanFilter(
            std_weight_position=0.1, std_weight_velocity=0.05
        )

        assert kf._std_weight_position == 0.1
        assert kf._std_weight_velocity == 0.05

    def test_basketball_params_higher_than_defaults(self):
        """Basketball-tuned params should be higher than boxmot defaults."""
        default_pos = 1 / 20  # boxmot default
        default_vel = 1 / 160  # boxmot default

        assert BASKETBALL_STD_WEIGHT_POSITION > default_pos
        assert BASKETBALL_STD_WEIGHT_VELOCITY > default_vel

    def test_motion_matrix_shape(self):
        """State transition matrix should be 8x8."""
        kf = BasketballKalmanFilter()
        assert kf._motion_mat.shape == (8, 8)

    def test_update_matrix_shape(self):
        """Measurement matrix should be 4x8."""
        kf = BasketballKalmanFilter()
        assert kf._update_mat.shape == (4, 8)


class TestKalmanFilterInitiate:
    """Tests for track initiation from first measurement."""

    def test_initiate_returns_correct_shapes(self):
        """Mean should be (8,) and covariance (8,8)."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])  # cx, cy, aspect, height

        mean, covariance = kf.initiate(measurement)

        assert mean.shape == (8,)
        assert covariance.shape == (8, 8)

    def test_initiate_position_matches_measurement(self):
        """Initial position should match the measurement."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])

        mean, _ = kf.initiate(measurement)

        np.testing.assert_array_almost_equal(mean[:4], measurement)

    def test_initiate_velocity_is_zero(self):
        """Initial velocity should be zero."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])

        mean, _ = kf.initiate(measurement)

        np.testing.assert_array_equal(mean[4:], 0)

    def test_initiate_covariance_is_diagonal(self):
        """Initial covariance should be diagonal."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])

        _, covariance = kf.initiate(measurement)

        # Check that off-diagonal elements are zero
        off_diag = covariance - np.diag(np.diag(covariance))
        np.testing.assert_array_almost_equal(off_diag, 0)

    def test_initiate_velocity_uncertainty_nonzero(self):
        """Velocity uncertainty should be nonzero (unknown initial velocity)."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])

        _, covariance = kf.initiate(measurement)
        diag = np.diag(covariance)

        # Velocity variances (indices 4,5,7) should be positive
        assert diag[4] > 0  # vx variance
        assert diag[5] > 0  # vy variance
        assert diag[7] > 0  # vh variance
        # Position variances should also be positive
        assert diag[0] > 0  # x variance
        assert diag[1] > 0  # y variance
        assert diag[3] > 0  # h variance


class TestKalmanFilterPredictUpdate:
    """Tests for the predict-update cycle."""

    def test_predict_advances_position(self):
        """Prediction should move position by velocity * dt."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])

        mean, covariance = kf.initiate(measurement)

        # Set initial velocity
        mean[4] = 10  # vx = 10 pixels/frame
        mean[5] = -5  # vy = -5 pixels/frame

        predicted_mean, _ = kf.predict(mean, covariance)

        assert predicted_mean[0] == pytest.approx(650, rel=1e-5)  # x + vx
        assert predicted_mean[1] == pytest.approx(355, rel=1e-5)  # y + vy

    def test_predict_increases_uncertainty(self):
        """Prediction should increase state uncertainty (covariance grows)."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])

        mean, covariance = kf.initiate(measurement)
        _, predicted_cov = kf.predict(mean, covariance)

        # Diagonal should be larger after prediction
        assert np.all(np.diag(predicted_cov) >= np.diag(covariance))

    def test_update_reduces_uncertainty(self):
        """Update with a measurement should reduce uncertainty."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])

        mean, covariance = kf.initiate(measurement)
        predicted_mean, predicted_cov = kf.predict(mean, covariance)

        # Update with same measurement
        updated_mean, updated_cov = kf.update(
            predicted_mean, predicted_cov, measurement
        )

        # Position uncertainty should decrease
        assert np.diag(updated_cov)[0] <= np.diag(predicted_cov)[0]

    def test_predict_update_cycle_converges(self):
        """Multiple cycles with consistent measurements should converge."""
        kf = BasketballKalmanFilter()

        # Simulate a player moving at constant velocity
        true_position = np.array([640, 360, 0.5, 200])
        velocity = np.array([5, 3, 0, 0])

        mean, covariance = kf.initiate(true_position)
        initial_cov_trace = np.trace(covariance)

        for i in range(20):
            mean, covariance = kf.predict(mean, covariance)
            measurement = true_position + velocity * (i + 1)
            measurement = measurement + np.random.randn(4) * 2  # Add noise
            mean, covariance = kf.update(mean, covariance, measurement)

        # Covariance should have decreased (more certain)
        final_cov_trace = np.trace(covariance)
        assert final_cov_trace < initial_cov_trace


class TestNSAKalman:
    """Tests for confidence-scaled measurement noise (NSA Kalman)."""

    def test_high_confidence_less_noise(self):
        """High confidence should produce lower measurement noise."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])
        mean, covariance = kf.initiate(measurement)

        _, cov_high = kf.project(mean, covariance, confidence=0.9)
        _, cov_low = kf.project(mean, covariance, confidence=0.1)

        # Higher confidence → lower projected covariance
        assert np.trace(cov_high) < np.trace(cov_low)

    def test_zero_confidence_no_scaling(self):
        """Zero confidence should use default measurement noise."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])
        mean, covariance = kf.initiate(measurement)

        _, cov_zero = kf.project(mean, covariance, confidence=0.0)
        _, cov_default = kf.project(mean, covariance)

        np.testing.assert_array_almost_equal(cov_zero, cov_default)


class TestGatingDistance:
    """Tests for Mahalanobis gating distance."""

    def test_same_measurement_zero_distance(self):
        """A measurement matching the projected mean should have near-zero distance."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])
        mean, covariance = kf.initiate(measurement)

        # The measurement should have very low gating distance
        distances = kf.gating_distance(
            mean, covariance, measurement.reshape(1, -1)
        )

        assert distances[0] < 1.0  # Should be very small

    def test_far_measurement_large_distance(self):
        """A measurement far from the mean should have a large distance."""
        kf = BasketballKalmanFilter()
        measurement = np.array([640, 360, 0.5, 200])
        mean, covariance = kf.initiate(measurement)

        far_measurement = np.array([[1000, 100, 0.8, 300]])
        distances = kf.gating_distance(mean, covariance, far_measurement)

        assert distances[0] > 10.0  # Should be large
