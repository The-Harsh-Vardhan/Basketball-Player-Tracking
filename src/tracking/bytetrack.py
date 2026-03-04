"""
ByteTrack wrapper with basketball-tuned Kalman Filter.

Wraps the boxmot ByteTrack implementation and patches it with
a custom Kalman Filter tuned for basketball player dynamics.

ByteTrack algorithm overview:
    1. Split detections into high-confidence and low-confidence groups.
    2. First association: match high-confidence detections to existing tracks
       using IoU distance.
    3. Second association: match remaining (unmatched) tracks against
       low-confidence detections — this rescues partially occluded players.
    4. Handle unmatched tracks: keep them alive for `track_buffer` frames
       using Kalman prediction.
    5. Initialize new tracks from unmatched high-confidence detections.

The two-stage association is critical for the "screen" edge case:
    When a player is partially occluded by a screen, their detection
    confidence drops. The second stage matches these low-confidence
    detections to the predicted track position, preserving the ID.
"""

import logging
from pathlib import Path

import numpy as np
import yaml

from src.tracking.kalman_filter import BasketballKalmanFilter

logger = logging.getLogger(__name__)

# Default tracker parameters (basketball-tuned)
DEFAULT_TRACKER_PARAMS = {
    "track_thresh": 0.5,
    "match_thresh": 0.8,
    "track_buffer": 75,
    "frame_rate": 25,
    "min_hits": 3,
}


class BasketballTracker:
    """
    ByteTrack-based tracker customized for basketball player tracking.

    Wraps boxmot's ByteTrack with:
        - Basketball-tuned Kalman Filter (higher position/velocity noise)
        - Extended track buffer for boundary re-entry (3s at 25 FPS)
        - Configurable parameters via YAML

    The tracker maintains a list of active tracks, each with a unique ID.
    On each frame, call `update()` with detections to get tracked outputs.

    Attributes:
        tracker: The underlying boxmot ByteTrack instance.
        kalman_filter: Basketball-tuned Kalman Filter.
        frame_count: Number of frames processed.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        **kwargs,
    ):
        """
        Initialize the basketball tracker.

        Args:
            config_path: Path to tracker.yaml config file.
                If provided, loads parameters from the 'tracker' section.
            **kwargs: Override any tracker parameter directly.
        """
        # Load configuration
        params = {**DEFAULT_TRACKER_PARAMS}
        if config_path:
            file_config = self._load_config(Path(config_path))
            params.update(file_config)
        params.update(kwargs)  # CLI/code overrides take priority

        self.params = params
        self.frame_count = 0

        # Initialize basketball-tuned Kalman Filter
        kf_params = params.get("kalman", {})
        self.kalman_filter = BasketballKalmanFilter(
            std_weight_position=kf_params.get("std_weight_position", 1.0 / 15),
            std_weight_velocity=kf_params.get("std_weight_velocity", 1.0 / 100),
        )

        # Initialize ByteTrack from boxmot
        self._init_tracker(params)

        logger.info(
            f"BasketballTracker initialized: "
            f"track_buffer={params['track_buffer']}, "
            f"track_thresh={params['track_thresh']}, "
            f"frame_rate={params['frame_rate']}"
        )

    def _load_config(self, config_path: Path) -> dict:
        """Load tracker parameters from YAML config."""
        if not config_path.is_file():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config.get("tracker", {})

    def _init_tracker(self, params: dict) -> None:
        """
        Initialize the boxmot ByteTrack instance and patch Kalman Filter.

        Monkey-patches the tracker's internal Kalman Filter with our
        basketball-tuned version.
        """
        from boxmot import ByteTrack

        self.tracker = ByteTrack(
            track_thresh=params.get("track_thresh", 0.5),
            match_thresh=params.get("match_thresh", 0.8),
            track_buffer=params.get("track_buffer", 75),
            frame_rate=params.get("frame_rate", 25),
        )

        # Patch the Kalman Filter in ByteTrack's internal tracks
        # ByteTrack uses KalmanFilterXYAH — we replace it with our tuned version
        self._patch_kalman_filter()

    def _patch_kalman_filter(self) -> None:
        """
        Replace ByteTrack's default Kalman Filter with basketball-tuned version.

        The boxmot ByteTrack stores a shared KF instance. We replace it
        with our BasketballKalmanFilter which has higher noise tolerances
        for the fast, jerk-heavy motion of basketball players.
        """
        try:
            # boxmot ByteTrack typically has a shared_kalman or per-track KF
            # We patch at the class level so all new tracks use our KF
            from boxmot.trackers.bytetrack.byte_tracker import STrack

            # Replace the shared Kalman filter
            if hasattr(STrack, "shared_kalman"):
                STrack.shared_kalman = self.kalman_filter
                logger.info("Patched STrack.shared_kalman with BasketballKF")
            else:
                logger.warning(
                    "STrack.shared_kalman not found — KF patch may not apply. "
                    "Check boxmot version compatibility."
                )
        except ImportError as e:
            logger.error(f"Failed to patch Kalman Filter: {e}")

    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Update tracker with new detections for the current frame.

        Args:
            detections: Detection array of shape (N, 6):
                Each row: [x1, y1, x2, y2, confidence, class_id]
                Coordinates are in pixel space (not normalized).
                If no detections, pass np.empty((0, 6)).
            frame: Current video frame as numpy array (H, W, C) BGR.
                Required by boxmot for potential ReID features.

        Returns:
            Tracked objects array of shape (M, 8):
                Each row: [x1, y1, x2, y2, track_id, confidence, class_id, det_index]
                Returns np.empty((0, 8)) if no active tracks.
        """
        self.frame_count += 1

        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6))

        results = self.tracker.update(detections, frame)

        if results is None or len(results) == 0:
            return np.empty((0, 8))

        return results

    def reset(self) -> None:
        """Reset the tracker state (clear all tracks)."""
        self.frame_count = 0
        self._init_tracker(self.params)
        logger.info("Tracker reset")

    @property
    def active_tracks(self) -> int:
        """Number of currently active (confirmed) tracks."""
        if hasattr(self.tracker, "tracked_stracks"):
            return len(self.tracker.tracked_stracks)
        return 0

    @property
    def lost_tracks(self) -> int:
        """Number of currently lost (unconfirmed) tracks."""
        if hasattr(self.tracker, "lost_stracks"):
            return len(self.tracker.lost_stracks)
        return 0
