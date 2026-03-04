"""
Video visualization for tracked basketball players.

Renders bounding boxes, track IDs, and confidence scores on video frames.
Supports both real-time display and batch rendering to .mp4 files.
"""

import colorsys
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Visual style constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
LABEL_PADDING = 5


class TrackVisualizer:
    """
    Renders tracking results on video frames.

    Each track ID is assigned a persistent, visually distinct color
    using HSV color space with golden ratio spacing.

    Attributes:
        color_map: Dictionary mapping track IDs to BGR colors.
    """

    def __init__(self, max_colors: int = 100):
        """
        Initialize the visualizer.

        Args:
            max_colors: Number of pre-generated colors in the palette.
        """
        self.color_map: dict[int, tuple[int, int, int]] = {}
        self._palette = self._generate_palette(max_colors)

    @staticmethod
    def _generate_palette(n: int) -> list[tuple[int, int, int]]:
        """
        Generate N visually distinct colors using golden ratio spacing in HSV.

        Args:
            n: Number of colors to generate.

        Returns:
            List of BGR color tuples.
        """
        palette = []
        golden_ratio = 0.618033988749895
        hue = 0.0

        for _ in range(n):
            hue = (hue + golden_ratio) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            # Convert to BGR (0-255)
            palette.append((int(b * 255), int(g * 255), int(r * 255)))

        return palette

    def _get_color(self, track_id: int) -> tuple[int, int, int]:
        """Get a persistent color for a track ID."""
        if track_id not in self.color_map:
            idx = len(self.color_map) % len(self._palette)
            self.color_map[track_id] = self._palette[idx]
        return self.color_map[track_id]

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        frame_idx: int | None = None,
        show_conf: bool = True,
        show_id: bool = True,
    ) -> np.ndarray:
        """
        Draw tracked bounding boxes on a frame.

        Args:
            frame: BGR image (H, W, 3). Will be copied, not modified in-place.
            tracks: Tracked objects array (M, 8):
                [x1, y1, x2, y2, track_id, conf, cls, det_index]
            frame_idx: Optional frame number to display in the corner.
            show_conf: Whether to display confidence scores.
            show_id: Whether to display track IDs.

        Returns:
            Annotated frame as numpy array.
        """
        annotated = frame.copy()

        if tracks is None or len(tracks) == 0:
            if frame_idx is not None:
                self._draw_frame_info(annotated, frame_idx, 0)
            return annotated

        for track in tracks:
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            track_id = int(track[4])
            conf = float(track[5])

            color = self._get_color(track_id)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # Build label text
            parts = []
            if show_id:
                parts.append(f"ID:{track_id}")
            if show_conf:
                parts.append(f"{conf:.2f}")
            label = " ".join(parts)

            if label:
                self._draw_label(annotated, label, x1, y1, color)

        # Draw frame info overlay
        if frame_idx is not None:
            self._draw_frame_info(annotated, frame_idx, len(tracks))

        return annotated

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a label with background above the bounding box."""
        (tw, th), baseline = cv2.getTextSize(
            text, FONT, FONT_SCALE, FONT_THICKNESS
        )
        # Label background
        label_y1 = max(0, y - th - 2 * LABEL_PADDING)
        label_y2 = y
        label_x1 = x
        label_x2 = x + tw + 2 * LABEL_PADDING

        cv2.rectangle(
            frame, (label_x1, label_y1), (label_x2, label_y2), color, -1
        )
        # Text (white or black depending on background brightness)
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (x + LABEL_PADDING, y - LABEL_PADDING),
            FONT,
            FONT_SCALE,
            text_color,
            FONT_THICKNESS,
        )

    @staticmethod
    def _draw_frame_info(
        frame: np.ndarray, frame_idx: int, num_tracks: int
    ) -> None:
        """Draw frame number and track count in the top-left corner."""
        info = f"Frame: {frame_idx} | Tracks: {num_tracks}"
        cv2.putText(
            frame, info, (10, 30), FONT, 0.7, (0, 255, 0), 2
        )


def create_video_from_images(
    image_dir,
    output_path,
    tracks_file=None,
    fps: int = 25,
) -> None:
    """
    Create an annotated .mp4 video from an image sequence and MOT results.

    Args:
        image_dir: Directory containing sequential frame images (.jpg).
        output_path: Path for the output .mp4 video.
        tracks_file: Optional path to MOT-format tracker_output.txt.
            If provided, draws tracked bounding boxes on frames.
        fps: Output video frame rate.
    """
    from pathlib import Path

    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    # Read first image for dimensions
    sample = cv2.imread(str(images[0]))
    height, width = sample.shape[:2]

    # Parse tracks if provided
    frame_tracks: dict[int, list] = {}
    if tracks_file:
        tracks_path = Path(tracks_file)
        if tracks_path.is_file():
            with open(tracks_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 7:
                        fid = int(parts[0])
                        tid = int(parts[1])
                        x, y, w, h = (
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                            float(parts[5]),
                        )
                        conf = float(parts[6])
                        track_data = [x, y, x + w, y + h, tid, conf, 0, 0]
                        frame_tracks.setdefault(fid, []).append(track_data)

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    visualizer = TrackVisualizer()

    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        if idx in frame_tracks:
            tracks_array = np.array(frame_tracks[idx])
            frame = visualizer.draw_tracks(frame, tracks_array, idx)
        else:
            frame = visualizer.draw_tracks(frame, np.empty((0, 8)), idx)

        writer.write(frame)

    writer.release()
    logger.info(f"Video saved: {output_path} ({len(images)} frames @ {fps}fps)")
