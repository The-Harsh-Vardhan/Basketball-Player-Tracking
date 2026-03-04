#!/usr/bin/env python3
"""
Inference script for the basketball player tracking pipeline.

Runs the end-to-end detection + tracking pipeline on a video or image sequence,
producing:
    - tracker_output.txt (MOT format)
    - output.mp4 (annotated video with bounding boxes, IDs, confidence)
    - Performance metrics (FPS, track count)

Usage:
    # Process a video file
    python scripts/run_inference.py --source video.mp4

    # Process a SportsMOT image sequence
    python scripts/run_inference.py --source data/SportsMOT/dataset/val/v_xxx/img1/

    # With custom model and config
    python scripts/run_inference.py --source video.mp4 --weights best.pt --config configs/tracker.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run basketball player tracking inference"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input source: video file (.mp4) or image directory",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_root / "runs" / "detect" / "yolo11m_basketball" / "weights" / "best.pt",
        help="Path to trained YOLO model weights",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "tracker.yaml",
        help="Path to tracker configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs",
        help="Directory for output files",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for YOLO",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for inference (0=GPU, cpu=CPU)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 inference",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip output video generation",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames during processing",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate inputs
    if not args.source.exists():
        logger.error(f"Source not found: {args.source}")
        sys.exit(1)

    if not args.weights.is_file():
        logger.error(f"Model weights not found: {args.weights}")
        logger.error(
            "Train the model first with: python scripts/train.py"
        )
        sys.exit(1)

    from src.inference.pipeline import TrackingPipeline

    # Initialize pipeline
    logger.info("=" * 60)
    logger.info("Basketball Player Tracking — Inference")
    logger.info("=" * 60)
    logger.info(f"Source:  {args.source}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Config:  {args.config}")
    logger.info(f"Device:  {args.device}")

    pipeline = TrackingPipeline(
        model_weights=args.weights,
        tracker_config=args.config if args.config.is_file() else None,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=not args.no_half,
    )

    # Prepare output paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mot_output = args.output_dir / "tracker_output.txt"
    video_output = args.output_dir / "output.mp4" if not args.no_video else None

    # Determine source type and process
    if args.source.is_dir():
        # Image sequence (e.g., SportsMOT img1/)
        logger.info("Processing image sequence...")
        metrics = pipeline.process_image_sequence(
            image_dir=args.source,
            output_mot_path=mot_output,
        )

        # Create video from image sequence + tracking results
        if video_output:
            from src.inference.visualize import create_video_from_images
            logger.info("Generating output video from image sequence...")
            create_video_from_images(
                image_dir=args.source,
                output_path=video_output,
                tracks_file=mot_output,
            )
            metrics["output_video"] = str(video_output)

    elif args.source.is_file():
        # Video file
        logger.info("Processing video file...")
        metrics = pipeline.process_video(
            video_path=args.source,
            output_mot_path=mot_output,
            output_video_path=video_output,
            show=args.show,
        )
    else:
        logger.error(f"Invalid source: {args.source}")
        sys.exit(1)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Inference Results")
    logger.info("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    # Check FPS requirement
    if metrics.get("avg_fps", 0) >= 30:
        logger.info(
            f"\n  ✓ FPS requirement met: {metrics['avg_fps']:.1f} FPS (>30 FPS)"
        )
    else:
        logger.warning(
            f"\n  ✗ FPS below target: {metrics.get('avg_fps', 0):.1f} FPS (<30 FPS)"
        )

    # Save metrics
    metrics_path = args.output_dir / "inference_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
