"""
End-to-end detection + tracking inference pipeline.

Processes video frames or image sequences through YOLO11 detection
and ByteTrack tracking, producing MOT-format output and FPS metrics.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class TrackingPipeline:
    """
    End-to-end object detection and tracking pipeline.

    Orchestrates YOLO11 detection → ByteTrack tracking → MOT output.

    Pipeline flow per frame:
        1. Read frame from video/image sequence
        2. Run YOLO11 inference → detections (x1, y1, x2, y2, conf, cls)
        3. Filter detections for target class (player = 0)
        4. Feed detections to ByteTrack → tracked objects with persistent IDs
        5. Accumulate results in MOT format
        6. Optionally render annotated frames

    Attributes:
        model: Ultralytics YOLO model for detection.
        tracker: BasketballTracker instance.
        results: List of MOT-format result strings.
    """

    def __init__(
        self,
        model_weights: str | Path,
        tracker_config: str | Path | None = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
        half: bool = True,
        target_class: int | None = None,
    ):
        """
        Initialize the tracking pipeline.

        Args:
            model_weights: Path to trained YOLO model weights (.pt).
            tracker_config: Path to tracker.yaml configuration file.
            conf_thresh: Detection confidence threshold.
            iou_thresh: NMS IoU threshold.
            imgsz: Input image size for YOLO.
            device: Device for inference ('0' for GPU, 'cpu' for CPU).
            half: Use FP16 inference for speed.
            target_class: If set, filter detections to this YOLO class index.
                For our basketball dataset (single-class), this is None (accept all).
        """
        from ultralytics import YOLO
        from src.tracking.bytetrack import BasketballTracker

        # Load detection model
        self.model_weights = Path(model_weights)
        if not self.model_weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {model_weights}")

        logger.info(f"Loading YOLO model: {model_weights}")
        self.model = YOLO(str(model_weights))

        # Detection parameters
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.target_class = target_class

        # Initialize tracker
        self.tracker = BasketballTracker(config_path=tracker_config)

        # Results accumulator
        self.results: list[str] = []
        self.frame_times: list[float] = []

    def process_video(
        self,
        video_path: str | Path,
        output_mot_path: str | Path | None = None,
        output_video_path: str | Path | None = None,
        show: bool = False,
    ) -> dict:
        """
        Process an entire video through the pipeline.

        Args:
            video_path: Path to input video file.
            output_mot_path: Path to write MOT-format results (tracker_output.txt).
            output_video_path: Path to write annotated output video (.mp4).
            show: Whether to display frames in a window during processing.

        Returns:
            Dictionary with pipeline metrics:
                - total_frames: Number of frames processed
                - avg_fps: Average frames per second
                - total_tracks: Number of unique track IDs
                - output_mot: Path to MOT output file (if saved)
                - output_video: Path to video output file (if saved)
        """
        from src.inference.visualize import TrackVisualizer

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Processing video: {video_path} "
            f"({width}x{height} @ {fps}fps, {total_frames} frames)"
        )

        # Initialize video writer if output requested
        writer = None
        if output_video_path:
            output_video_path = Path(output_video_path)
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )

        visualizer = TrackVisualizer()
        self.results = []
        self.frame_times = []
        self.tracker.reset()
        frame_idx = 0
        unique_ids: set[int] = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_start = time.perf_counter()

            # Run detection + tracking on single frame
            tracked = self._process_frame(frame)
            t_end = time.perf_counter()
            self.frame_times.append(t_end - t_start)

            # Accumulate MOT results
            for track in tracked:
                x1, y1, x2, y2, track_id, conf = (
                    track[0], track[1], track[2], track[3],
                    int(track[4]), track[5],
                )
                bb_left = x1
                bb_top = y1
                bb_width = x2 - x1
                bb_height = y2 - y1
                unique_ids.add(track_id)

                mot_line = (
                    f"{frame_idx},{track_id},"
                    f"{bb_left:.2f},{bb_top:.2f},"
                    f"{bb_width:.2f},{bb_height:.2f},"
                    f"{conf:.4f},-1,-1,-1"
                )
                self.results.append(mot_line)

            # Visualize
            if writer or show:
                annotated = visualizer.draw_tracks(frame, tracked, frame_idx)
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Basketball Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Progress logging
            if frame_idx % 100 == 0:
                avg_fps = frame_idx / sum(self.frame_times)
                logger.info(
                    f"Frame {frame_idx}/{total_frames} | "
                    f"FPS: {avg_fps:.1f} | "
                    f"Active tracks: {self.tracker.active_tracks} | "
                    f"Lost: {self.tracker.lost_tracks}"
                )

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        # Save MOT results
        if output_mot_path:
            output_mot_path = Path(output_mot_path)
            output_mot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_mot_path, "w") as f:
                f.write("\n".join(self.results))
                if self.results:
                    f.write("\n")
            logger.info(f"MOT results saved to: {output_mot_path}")

        # Compute metrics
        avg_fps = (
            len(self.frame_times) / sum(self.frame_times)
            if self.frame_times
            else 0
        )

        metrics = {
            "total_frames": frame_idx,
            "avg_fps": avg_fps,
            "total_tracks": len(unique_ids),
            "output_mot": str(output_mot_path) if output_mot_path else None,
            "output_video": str(output_video_path) if output_video_path else None,
        }

        logger.info(
            f"Pipeline complete: {frame_idx} frames, "
            f"{avg_fps:.1f} FPS, {len(unique_ids)} unique tracks"
        )

        return metrics

    def process_image_sequence(
        self,
        image_dir: str | Path,
        output_mot_path: str | Path | None = None,
        frame_rate: int = 25,
    ) -> dict:
        """
        Process an image sequence (e.g., SportsMOT img1/ directory).

        Args:
            image_dir: Directory containing sequential .jpg frames.
            output_mot_path: Path to write MOT-format results.
            frame_rate: Frame rate for FPS calculations.

        Returns:
            Dictionary with pipeline metrics.
        """
        image_dir = Path(image_dir)
        images = sorted(image_dir.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No .jpg images in: {image_dir}")

        logger.info(f"Processing {len(images)} images from: {image_dir}")

        self.results = []
        self.frame_times = []
        self.tracker.reset()
        unique_ids: set[int] = set()

        for frame_idx, img_path in enumerate(images, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"Failed to read: {img_path}")
                continue

            t_start = time.perf_counter()
            tracked = self._process_frame(frame)
            t_end = time.perf_counter()
            self.frame_times.append(t_end - t_start)

            for track in tracked:
                x1, y1, x2, y2, track_id, conf = (
                    track[0], track[1], track[2], track[3],
                    int(track[4]), track[5],
                )
                unique_ids.add(track_id)
                mot_line = (
                    f"{frame_idx},{track_id},"
                    f"{x1:.2f},{y1:.2f},"
                    f"{x2 - x1:.2f},{y2 - y1:.2f},"
                    f"{conf:.4f},-1,-1,-1"
                )
                self.results.append(mot_line)

        # Save results
        if output_mot_path:
            output_mot_path = Path(output_mot_path)
            output_mot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_mot_path, "w") as f:
                f.write("\n".join(self.results))
                if self.results:
                    f.write("\n")

        avg_fps = (
            len(self.frame_times) / sum(self.frame_times)
            if self.frame_times
            else 0
        )

        return {
            "total_frames": len(images),
            "avg_fps": avg_fps,
            "total_tracks": len(unique_ids),
            "output_mot": str(output_mot_path) if output_mot_path else None,
        }

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection and tracking on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            Tracked objects array (M, 8):
                [x1, y1, x2, y2, track_id, conf, cls, det_index]
        """
        # YOLO detection
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        # Extract detections as numpy array
        if len(results.boxes):
            dets = np.hstack([
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy().reshape(-1, 1),
                results.boxes.cls.cpu().numpy().reshape(-1, 1),
            ])

            # Filter by target class if specified
            if self.target_class is not None:
                mask = dets[:, 5].astype(int) == self.target_class
                dets = dets[mask]
        else:
            dets = np.empty((0, 6))

        # Update tracker
        tracked = self.tracker.update(dets, frame)

        return tracked


def load_pipeline_config(config_path: str | Path) -> dict:
    """
    Load pipeline configuration from tracker.yaml.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with inference settings.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("inference", {})
