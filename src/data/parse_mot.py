"""
Parse MOT Challenge ground truth annotations into normalized YOLO format.

Converts gt.txt files (MOT format) into per-frame YOLO annotation files.

MOT format (gt.txt):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    Each field is comma-separated, 1-indexed frame numbers.

YOLO format (per-frame .txt):
    class_id cx cy w h
    Values are normalized to [0, 1] by image width/height.
    One line per object in the frame.

Image dimensions are read from seqinfo.ini for each sequence.
"""

import configparser
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SeqInfo:
    """Metadata parsed from seqinfo.ini."""

    name: str
    img_dir: str
    frame_rate: int
    seq_length: int
    img_width: int
    img_height: int
    img_ext: str


@dataclass
class MOTAnnotation:
    """Single MOT ground truth annotation row."""

    frame: int
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    conf: float
    cls: int
    visibility: float


def parse_seqinfo(seqinfo_path: Path) -> SeqInfo:
    """
    Parse a seqinfo.ini file to extract sequence metadata.

    Args:
        seqinfo_path: Path to the seqinfo.ini file.

    Returns:
        SeqInfo dataclass with parsed fields.

    Example seqinfo.ini:
        [Sequence]
        name=v_-6Os86HzwCs_c001
        imDir=img1
        frameRate=25
        seqLength=825
        imWidth=1280
        imHeight=720
        imExt=.jpg
    """
    if not seqinfo_path.is_file():
        raise FileNotFoundError(f"seqinfo.ini not found: {seqinfo_path}")

    config = configparser.ConfigParser()
    config.read(str(seqinfo_path))

    section = "Sequence"
    return SeqInfo(
        name=config.get(section, "name"),
        img_dir=config.get(section, "imDir"),
        frame_rate=config.getint(section, "frameRate"),
        seq_length=config.getint(section, "seqLength"),
        img_width=config.getint(section, "imWidth"),
        img_height=config.getint(section, "imHeight"),
        img_ext=config.get(section, "imExt"),
    )


def parse_gt_file(gt_path: Path) -> list[MOTAnnotation]:
    """
    Parse a MOT Challenge gt.txt file into a list of annotations.

    Args:
        gt_path: Path to gt.txt.

    Returns:
        List of MOTAnnotation objects.

    Raises:
        FileNotFoundError: If gt.txt does not exist.
        ValueError: If a line cannot be parsed.
    """
    if not gt_path.is_file():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    annotations = []
    with open(gt_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 9:
                logger.warning(
                    f"Skipping malformed line {line_num} in {gt_path}: {line}"
                )
                continue

            try:
                ann = MOTAnnotation(
                    frame=int(parts[0]),
                    track_id=int(parts[1]),
                    bb_left=float(parts[2]),
                    bb_top=float(parts[3]),
                    bb_width=float(parts[4]),
                    bb_height=float(parts[5]),
                    conf=float(parts[6]),
                    cls=int(parts[7]),
                    visibility=float(parts[8]),
                )
                annotations.append(ann)
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Error parsing line {line_num} in {gt_path}: {e}"
                )

    logger.info(f"Parsed {len(annotations)} annotations from {gt_path.name}")
    return annotations


def group_annotations_by_frame(
    annotations: list[MOTAnnotation],
) -> dict[int, list[MOTAnnotation]]:
    """
    Group annotations by frame number.

    Args:
        annotations: List of MOTAnnotation objects.

    Returns:
        Dictionary mapping frame number to list of annotations in that frame.
    """
    frame_groups: dict[int, list[MOTAnnotation]] = {}
    for ann in annotations:
        frame_groups.setdefault(ann.frame, []).append(ann)
    return frame_groups


def mot_to_yolo(
    ann: MOTAnnotation, img_width: int, img_height: int
) -> str:
    """
    Convert a single MOT annotation to a YOLO format line.

    Conversion:
        MOT: (bb_left, bb_top, bb_width, bb_height) in pixels
        YOLO: (class_id, cx, cy, w, h) normalized to [0, 1]

    The class_id is always 0 (player) for this single-class task.

    Args:
        ann: MOT annotation to convert.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        YOLO format string: "0 cx cy w h"
    """
    # Compute normalized center coordinates and dimensions
    cx = (ann.bb_left + ann.bb_width / 2.0) / img_width
    cy = (ann.bb_top + ann.bb_height / 2.0) / img_height
    w = ann.bb_width / img_width
    h = ann.bb_height / img_height

    # Clamp values to [0, 1] to handle edge cases
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_sequence(
    seq_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    prefix: str = "",
    create_symlinks: bool = False,
) -> int:
    """
    Convert a single SportsMOT sequence to YOLO format.

    For each frame in the sequence:
      - Copies (or symlinks) the image to output_images_dir
      - Creates a corresponding .txt label file in output_labels_dir
      - Empty frames get an empty .txt file (negative sample)

    File naming uses a prefix to avoid collisions between sequences:
        {prefix}_{frame_number}.jpg / .txt

    Args:
        seq_dir: Path to the sequence directory (contains gt/, img1/, seqinfo.ini).
        output_images_dir: Directory to write/link image files.
        output_labels_dir: Directory to write YOLO label files.
        prefix: Prefix for output filenames (typically the sequence name).
        create_symlinks: If True, create symlinks to images instead of copying.

    Returns:
        Number of frames processed.
    """
    import shutil

    # Parse sequence metadata
    seq_info = parse_seqinfo(seq_dir / "seqinfo.ini")
    prefix = prefix or seq_info.name

    # Parse ground truth
    gt_path = seq_dir / "gt" / "gt.txt"
    annotations = parse_gt_file(gt_path)
    frame_groups = group_annotations_by_frame(annotations)

    # Ensure output directories exist
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    img_dir = seq_dir / seq_info.img_dir
    frames_processed = 0

    for frame_idx in range(1, seq_info.seq_length + 1):
        # Source image path
        img_filename = f"{frame_idx:06d}{seq_info.img_ext}"
        src_img = img_dir / img_filename

        if not src_img.is_file():
            logger.warning(f"Missing image: {src_img}")
            continue

        # Output filenames with sequence prefix to avoid collisions
        out_name = f"{prefix}_{frame_idx:06d}"
        dst_img = output_images_dir / f"{out_name}{seq_info.img_ext}"
        dst_label = output_labels_dir / f"{out_name}.txt"

        # Copy or symlink image
        if create_symlinks:
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
        else:
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

        # Write YOLO label file
        frame_anns = frame_groups.get(frame_idx, [])
        yolo_lines = [
            mot_to_yolo(ann, seq_info.img_width, seq_info.img_height)
            for ann in frame_anns
        ]

        with open(dst_label, "w") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        frames_processed += 1

    logger.info(
        f"Converted {seq_info.name}: {frames_processed} frames, "
        f"{len(annotations)} annotations"
    )
    return frames_processed


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print(
            "Usage: python parse_mot.py <seq_dir> <output_images> <output_labels>"
        )
        sys.exit(1)

    seq_dir = Path(sys.argv[1])
    out_images = Path(sys.argv[2])
    out_labels = Path(sys.argv[3])

    n = convert_sequence(seq_dir, out_images, out_labels)
    print(f"Processed {n} frames")
