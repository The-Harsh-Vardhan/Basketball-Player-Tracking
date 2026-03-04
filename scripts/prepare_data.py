#!/usr/bin/env python3
"""
Data preparation script for the Basketball Player Tracking pipeline.

Orchestrates the full data pipeline:
    1. Validate the SportsMOT dataset is downloaded
    2. Filter basketball-only sequences
    3. Convert MOT annotations to YOLO format
    4. Organize into train/val splits with YOLO directory structure

Usage:
    python scripts/prepare_data.py [--dataset-root data/SportsMOT] [--output-dir data/basketball_yolo] [--symlink]

Output structure:
    data/basketball_yolo/
    ├── images/
    │   ├── train/    # {sequence}_{frame:06d}.jpg
    │   └── val/
    └── labels/
        ├── train/    # {sequence}_{frame:06d}.txt (YOLO format)
        └── val/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import validate_dataset, print_download_instructions
from src.data.filter_basketball import (
    get_basketball_sequences,
    get_sequence_dir,
    validate_sequence,
)
from src.data.parse_mot import convert_sequence

logger = logging.getLogger(__name__)


def prepare_basketball_dataset(
    dataset_root: Path,
    output_dir: Path,
    use_symlinks: bool = False,
) -> dict:
    """
    Prepare the basketball YOLO dataset from SportsMOT.

    Args:
        dataset_root: Path to the SportsMOT root directory.
        output_dir: Path to write the YOLO-formatted dataset.
        use_symlinks: If True, symlink images instead of copying (saves disk space).

    Returns:
        Dictionary with preparation statistics.
    """
    # Step 1: Validate dataset
    logger.info("Step 1: Validating SportsMOT dataset...")
    if not validate_dataset(dataset_root):
        print_download_instructions()
        raise RuntimeError(
            f"SportsMOT dataset not found or invalid at: {dataset_root}"
        )

    # Step 2: Filter basketball sequences
    logger.info("Step 2: Filtering basketball sequences...")
    splits_dir = dataset_root / "splits_txt"
    basketball_seqs = get_basketball_sequences(splits_dir)

    stats = {
        "train_sequences": len(basketball_seqs.get("train", [])),
        "val_sequences": len(basketball_seqs.get("val", [])),
        "train_frames": 0,
        "val_frames": 0,
        "skipped": 0,
    }

    # Step 3: Convert each sequence to YOLO format
    logger.info("Step 3: Converting MOT annotations to YOLO format...")

    for split in ["train", "val"]:
        sequences = basketball_seqs.get(split, [])
        if not sequences:
            logger.warning(f"No basketball sequences found for split: {split}")
            continue

        output_images = output_dir / "images" / split
        output_labels = output_dir / "labels" / split

        logger.info(f"Processing {split} split: {len(sequences)} sequences")

        for i, seq_name in enumerate(sequences, 1):
            try:
                seq_dir = get_sequence_dir(dataset_root, split, seq_name)

                if not validate_sequence(seq_dir, require_gt=True):
                    logger.warning(f"Skipping invalid sequence: {seq_name}")
                    stats["skipped"] += 1
                    continue

                frames = convert_sequence(
                    seq_dir=seq_dir,
                    output_images_dir=output_images,
                    output_labels_dir=output_labels,
                    prefix=seq_name,
                    create_symlinks=use_symlinks,
                )

                stats[f"{split}_frames"] += frames

                if i % 10 == 0 or i == len(sequences):
                    logger.info(
                        f"  [{split}] {i}/{len(sequences)} sequences processed"
                    )

            except Exception as e:
                logger.error(f"Error processing {seq_name}: {e}")
                stats["skipped"] += 1

    # Step 4: Update dataset.yaml with absolute paths
    logger.info("Step 4: Updating dataset configuration...")
    _update_dataset_yaml(project_root / "configs" / "dataset.yaml", output_dir)

    return stats


def _update_dataset_yaml(yaml_path: Path, dataset_dir: Path) -> None:
    """
    Update the dataset.yaml file with the correct absolute path.

    Args:
        yaml_path: Path to configs/dataset.yaml.
        dataset_dir: Absolute path to the YOLO dataset directory.
    """
    import yaml

    if not yaml_path.is_file():
        logger.warning(f"dataset.yaml not found at: {yaml_path}")
        return

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config["path"] = str(dataset_dir.resolve())

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated dataset.yaml path to: {dataset_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare basketball YOLO dataset from SportsMOT"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/SportsMOT"),
        help="Path to the SportsMOT root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/basketball_yolo"),
        help="Output directory for the YOLO-formatted dataset",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks for images instead of copying (saves disk space)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Basketball YOLO Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Output dir:   {args.output_dir}")
    logger.info(f"Symlinks:     {args.symlink}")

    try:
        stats = prepare_basketball_dataset(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            use_symlinks=args.symlink,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Dataset Preparation Complete")
        logger.info("=" * 60)
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
