#!/usr/bin/env python3
"""
Evaluation script for the basketball player tracking pipeline.

Computes MOTA, IDF1, and other MOT metrics by comparing tracker output
against ground truth annotations from SportsMOT validation sequences.

Usage:
    python scripts/evaluate.py --gt-dir data/SportsMOT/dataset/val --pred-dir outputs/predictions
    python scripts/evaluate.py --gt-file path/to/gt.txt --pred-file path/to/tracker_output.txt

The script can evaluate:
    - A single sequence (--gt-file + --pred-file)
    - Multiple sequences (--gt-dir + --pred-dir + --sequences)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import (
    evaluate_sequence,
    evaluate_multiple_sequences,
    print_metrics_table,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MOT tracking performance (MOTA, IDF1)"
    )

    # Single sequence evaluation
    parser.add_argument(
        "--gt-file",
        type=Path,
        default=None,
        help="Path to single ground truth file (MOT format)",
    )
    parser.add_argument(
        "--pred-file",
        type=Path,
        default=None,
        help="Path to single prediction file (MOT format)",
    )

    # Multi-sequence evaluation
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Root directory containing GT sequences (each with gt/gt.txt)",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Directory containing prediction files ({seq_name}.txt)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="List of sequence names to evaluate (if not specified, evaluates all)",
    )

    # Options
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine evaluation mode
    if args.gt_file and args.pred_file:
        # Single sequence evaluation
        logger.info("Evaluating single sequence...")
        logger.info(f"GT:   {args.gt_file}")
        logger.info(f"Pred: {args.pred_file}")

        results = evaluate_sequence(
            args.gt_file, args.pred_file, args.iou_threshold
        )
        print_metrics_table(results)

    elif args.gt_dir and args.pred_dir:
        # Multi-sequence evaluation
        if not args.gt_dir.is_dir():
            logger.error(f"GT directory not found: {args.gt_dir}")
            sys.exit(1)

        if not args.pred_dir.is_dir():
            logger.error(f"Prediction directory not found: {args.pred_dir}")
            sys.exit(1)

        # Auto-detect sequence names if not specified
        if args.sequences is None:
            sequences = [
                d.name for d in args.gt_dir.iterdir()
                if d.is_dir() and (d / "gt" / "gt.txt").is_file()
            ]
            sequences = sorted(sequences)
        else:
            sequences = args.sequences

        if not sequences:
            logger.error("No sequences found to evaluate")
            sys.exit(1)

        logger.info(f"Evaluating {len(sequences)} sequences...")
        results = evaluate_multiple_sequences(
            args.gt_dir, args.pred_dir, sequences, args.iou_threshold
        )
        print_metrics_table(results)

    else:
        parser.error(
            "Specify either (--gt-file + --pred-file) for single sequence, "
            "or (--gt-dir + --pred-dir) for multi-sequence evaluation"
        )
        sys.exit(1)

    # Save results as JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output_json}")

    # Log to W&B
    if args.wandb:
        try:
            import wandb

            wandb.init(
                project="basketball-player-tracking",
                job_type="evaluation",
            )
            if "aggregate" in results:
                wandb.log(results["aggregate"])
            else:
                wandb.log(results)
            wandb.finish()
            logger.info("Results logged to W&B")
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")


if __name__ == "__main__":
    main()
