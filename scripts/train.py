#!/usr/bin/env python3
"""
Training script for the YOLO11 basketball player detection model.

Fine-tunes a pretrained YOLO11m model on the SportsMOT basketball subset
with Weights & Biases integration for experiment tracking.

Usage:
    python scripts/train.py [--config configs/dataset.yaml] [--model yolo11m.pt] [--epochs 100]

Prerequisites:
    - Run `python scripts/prepare_data.py` first to prepare the dataset
    - Set WANDB_API_KEY in .env or environment
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 model for basketball player detection"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "dataset.yaml",
        help="Path to YOLO dataset.yaml config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="Pretrained model name or path (e.g., yolo11m.pt, yolo11s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (use -1 for auto-batch)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11m_basketball",
        help="Run name for W&B and output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for training (0, 1, cpu, etc.)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Path to weights — run validation only, skip training",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load .env file if present
    env_file = project_root / ".env"
    if env_file.is_file():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    from src.detection.train import YOLOTrainer

    # Validation-only mode
    if args.validate_only:
        logger.info(f"Running validation with weights: {args.validate_only}")
        trainer = YOLOTrainer(
            data_config=args.config,
            model_name=args.validate_only,
            wandb_enabled=not args.no_wandb,
        )
        metrics = trainer.validate()
        logger.info(
            f"Validation mAP@50: {metrics.box.map50:.4f}, "
            f"mAP@50:95: {metrics.box.map:.4f}"
        )
        return

    # Training
    logger.info("=" * 60)
    logger.info("YOLO11 Basketball Player Detection — Training")
    logger.info("=" * 60)
    logger.info(f"Model:   {args.model}")
    logger.info(f"Config:  {args.config}")
    logger.info(f"Epochs:  {args.epochs}")
    logger.info(f"ImgSz:   {args.imgsz}")
    logger.info(f"Batch:   {args.batch}")
    logger.info(f"Device:  {args.device}")
    logger.info(f"W&B:     {'disabled' if args.no_wandb else 'enabled'}")

    trainer = YOLOTrainer(
        data_config=args.config,
        model_name=args.model,
        project_dir=project_root / "runs" / "detect",
        wandb_enabled=not args.no_wandb,
    )

    train_kwargs = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
    }
    if args.resume:
        train_kwargs["resume"] = True

    results = trainer.train(run_name=args.name, **train_kwargs)

    # Run final validation
    best_weights = project_root / "runs" / "detect" / args.name / "weights" / "best.pt"
    if best_weights.is_file():
        logger.info(f"\nRunning final validation with best weights: {best_weights}")
        trainer.validate(weights_path=best_weights)
    else:
        logger.warning(f"Best weights not found at: {best_weights}")

    logger.info("\nTraining complete! Check W&B dashboard for detailed results.")


if __name__ == "__main__":
    main()
