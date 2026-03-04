"""
Filter SportsMOT dataset to isolate basketball-only sequences.

Reads the splits_txt/ metadata to identify basketball sequences within
each data split (train/val/test) and organizes them for downstream processing.

SportsMOT splits_txt/ structure:
    basketball.txt  — list of all basketball sequence names
    volleyball.txt  — list of all volleyball sequence names
    football.txt    — list of all football sequence names
    train.txt       — list of all training sequence names
    val.txt         — list of all validation sequence names
    test.txt        — list of all test sequence names

A basketball training sequence is one that appears in BOTH basketball.txt AND train.txt.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_sequence_list(filepath: Path) -> set[str]:
    """
    Read a sequence list file and return a set of sequence names.

    Each line in the file is a sequence name (e.g., 'v_-6Os86HzwCs_c001').
    Blank lines and whitespace are stripped.

    Args:
        filepath: Path to the .txt split file.

    Returns:
        Set of sequence name strings.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Split file not found: {filepath}")

    with open(filepath, "r") as f:
        sequences = {line.strip() for line in f if line.strip()}

    logger.info(f"Read {len(sequences)} sequences from {filepath.name}")
    return sequences


def get_basketball_sequences(
    splits_dir: Path,
) -> dict[str, list[str]]:
    """
    Identify basketball sequences for each data split.

    Cross-references basketball.txt with train.txt/val.txt/test.txt to produce
    per-split lists of basketball-only sequence names.

    Args:
        splits_dir: Path to the splits_txt/ directory.

    Returns:
        Dictionary mapping split names ('train', 'val', 'test') to lists of
        basketball sequence names within that split.

    Example:
        >>> seqs = get_basketball_sequences(Path("data/SportsMOT/splits_txt"))
        >>> print(len(seqs['train']))  # e.g., 45
    """
    basketball_seqs = _read_sequence_list(splits_dir / "basketball.txt")

    result = {}
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}.txt"
        if split_file.is_file():
            split_seqs = _read_sequence_list(split_file)
            basketball_in_split = sorted(basketball_seqs & split_seqs)
            result[split] = basketball_in_split
            logger.info(
                f"Basketball {split}: {len(basketball_in_split)} sequences"
            )
        else:
            logger.warning(f"Split file not found: {split_file}, skipping")
            result[split] = []

    return result


def get_sequence_dir(
    dataset_root: Path, split: str, sequence_name: str
) -> Path:
    """
    Get the directory path for a specific sequence.

    Args:
        dataset_root: Path to the SportsMOT root directory.
        split: Data split ('train', 'val', 'test').
        sequence_name: Name of the sequence.

    Returns:
        Path to the sequence directory.

    Raises:
        FileNotFoundError: If the sequence directory does not exist.
    """
    seq_dir = dataset_root / "dataset" / split / sequence_name
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    return seq_dir


def validate_sequence(seq_dir: Path, require_gt: bool = True) -> bool:
    """
    Validate that a sequence directory has the expected structure.

    Args:
        seq_dir: Path to a sequence directory.
        require_gt: Whether to require gt/gt.txt (False for test split).

    Returns:
        True if valid, False otherwise.
    """
    # Must have img1/ with at least one image
    img_dir = seq_dir / "img1"
    if not img_dir.is_dir():
        logger.warning(f"Missing img1/ in {seq_dir.name}")
        return False

    images = list(img_dir.glob("*.jpg"))
    if not images:
        logger.warning(f"No .jpg images in {seq_dir.name}/img1/")
        return False

    # Must have seqinfo.ini
    if not (seq_dir / "seqinfo.ini").is_file():
        logger.warning(f"Missing seqinfo.ini in {seq_dir.name}")
        return False

    # Must have gt/gt.txt (unless test split)
    if require_gt and not (seq_dir / "gt" / "gt.txt").is_file():
        logger.warning(f"Missing gt/gt.txt in {seq_dir.name}")
        return False

    return True


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    splits_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/SportsMOT/splits_txt"
    )

    basketball_seqs = get_basketball_sequences(splits_dir)
    for split, seqs in basketball_seqs.items():
        print(f"\n{split.upper()} ({len(seqs)} sequences):")
        for s in seqs[:5]:
            print(f"  {s}")
        if len(seqs) > 5:
            print(f"  ... and {len(seqs) - 5} more")
