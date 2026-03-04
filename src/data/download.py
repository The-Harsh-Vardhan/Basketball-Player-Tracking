"""
Download and setup instructions for the SportsMOT dataset.

The SportsMOT dataset is hosted on CodaLab and must be downloaded manually.
This module provides helper functions and instructions for dataset acquisition.

Dataset: SportsMOT (ICCV 2023)
    - 240 video clips across basketball, volleyball, and football
    - Resolution: 1280x720, 25 FPS
    - MOT Challenge annotation format (gt.txt)
    - Source: https://codalab.lisn.upsaclay.fr/competitions/12424

Expected directory structure after download:
    data/SportsMOT/
    ├── splits_txt/
    │   ├── basketball.txt
    │   ├── volleyball.txt
    │   ├── football.txt
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    └── dataset/
        ├── train/
        │   └── <sequence_name>/
        │       ├── gt/gt.txt
        │       ├── img1/000001.jpg ...
        │       └── seqinfo.ini
        ├── val/
        │   └── <same structure>
        └── test/
            └── <sequence_name>/
                ├── img1/000001.jpg ...
                └── seqinfo.ini
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Expected dataset structure markers
REQUIRED_SUBDIRS = ["splits_txt", "dataset"]
REQUIRED_SPLIT_FILES = [
    "basketball.txt", "volleyball.txt", "football.txt",
    "train.txt", "val.txt", "test.txt",
]


def validate_dataset(dataset_root: Path) -> bool:
    """
    Validate that the SportsMOT dataset is correctly downloaded and structured.

    Args:
        dataset_root: Path to the SportsMOT root directory.

    Returns:
        True if the dataset structure is valid, False otherwise.
    """
    if not dataset_root.exists():
        logger.error(f"Dataset root does not exist: {dataset_root}")
        return False

    # Check top-level directories
    for subdir in REQUIRED_SUBDIRS:
        if not (dataset_root / subdir).is_dir():
            logger.error(f"Missing required directory: {dataset_root / subdir}")
            return False

    # Check split files
    splits_dir = dataset_root / "splits_txt"
    for split_file in REQUIRED_SPLIT_FILES:
        if not (splits_dir / split_file).is_file():
            logger.error(f"Missing split file: {splits_dir / split_file}")
            return False

    # Check dataset splits
    dataset_dir = dataset_root / "dataset"
    for split in ["train", "val"]:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            logger.error(f"Missing dataset split directory: {split_dir}")
            return False
        # Check at least one sequence exists
        sequences = list(split_dir.iterdir())
        if not sequences:
            logger.error(f"No sequences found in: {split_dir}")
            return False

    logger.info(f"Dataset validation passed: {dataset_root}")
    return True


def print_download_instructions() -> None:
    """Print step-by-step instructions for downloading SportsMOT."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           SportsMOT Dataset Download Instructions           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  1. Visit: https://codalab.lisn.upsaclay.fr/competitions/12424
    ║  2. Sign up / Log in to CodaLab                              ║
    ║  3. Navigate to Participate → Get Data                       ║
    ║  4. Download the dataset archive                             ║
    ║  5. Extract to: data/SportsMOT/                              ║
    ║                                                              ║
    ║  Alternative download links:                                 ║
    ║  • OneDrive: https://1drv.ms/u/s!AtjeLq7YnYGRgQRrmqGr4B-k-xsC
    ║  • GitHub:   https://github.com/MCG-NJU/SportsMOT           ║
    ║                                                              ║
    ║  Expected structure after extraction:                        ║
    ║    data/SportsMOT/                                           ║
    ║    ├── splits_txt/                                           ║
    ║    │   ├── basketball.txt                                    ║
    ║    │   ├── train.txt                                         ║
    ║    │   └── ...                                               ║
    ║    └── dataset/                                              ║
    ║        ├── train/<seq>/gt/gt.txt, img1/, seqinfo.ini         ║
    ║        └── val/<seq>/gt/gt.txt, img1/, seqinfo.ini           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(instructions)


def get_dataset_stats(dataset_root: Path) -> dict:
    """
    Compute basic statistics about the downloaded dataset.

    Args:
        dataset_root: Path to the SportsMOT root directory.

    Returns:
        Dictionary with dataset statistics.
    """
    stats = {"train_sequences": 0, "val_sequences": 0, "test_sequences": 0}
    dataset_dir = dataset_root / "dataset"

    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if split_dir.is_dir():
            stats[f"{split}_sequences"] = len(
                [d for d in split_dir.iterdir() if d.is_dir()]
            )

    return stats


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/SportsMOT")

    if validate_dataset(dataset_path):
        stats = get_dataset_stats(dataset_path)
        print(f"Dataset stats: {stats}")
    else:
        print_download_instructions()
