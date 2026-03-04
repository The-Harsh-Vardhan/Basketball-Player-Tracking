"""
MOT evaluation metrics: MOTA, IDF1, and supporting computations.

Uses the py-motmetrics library to compute standard Multi-Object Tracking
evaluation metrics by comparing tracker output against ground truth.

Key metrics:
    MOTA (Multi-Object Tracking Accuracy):
        Combines false positives, false negatives, and ID switches
        into a single accuracy score. Range: (-inf, 1]. Higher is better.
        MOTA = 1 - (FN + FP + IDSW) / GT

    IDF1 (ID F-Score):
        Measures identity preservation quality — how well the tracker
        maintains consistent IDs over time. Range: [0, 1]. Higher is better.
        Harmonic mean of ID Precision and ID Recall.

    MOTP (Multi-Object Tracking Precision):
        Average overlap between matched predictions and ground truth.
        Measures localization quality. Range: [0, 1]. Higher is better.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_mot_file(filepath: Path) -> dict[int, list[dict]]:
    """
    Load a MOT Challenge format file (gt.txt or tracker_output.txt).

    MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

    Args:
        filepath: Path to a MOT-format text file.

    Returns:
        Dictionary mapping frame number to list of annotation dicts.
        Each dict has keys: 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"MOT file not found: {filepath}")

    frames: dict[int, list[dict]] = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            obj = {
                "id": int(parts[1]),
                "bb_left": float(parts[2]),
                "bb_top": float(parts[3]),
                "bb_width": float(parts[4]),
                "bb_height": float(parts[5]),
                "conf": float(parts[6]),
            }
            frames.setdefault(frame_id, []).append(obj)

    return frames


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes in (left, top, width, height) format.

    Args:
        box1: First box [left, top, width, height].
        box2: Second box [left, top, width, height].

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise IoU matrix between GT and predicted boxes.

    Args:
        gt_boxes: Ground truth boxes (N, 4) in [left, top, width, height].
        pred_boxes: Predicted boxes (M, 4) in [left, top, width, height].

    Returns:
        IoU matrix of shape (N, M).
    """
    n = len(gt_boxes)
    m = len(pred_boxes)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])

    return iou_matrix


def evaluate_sequence(
    gt_path: Path,
    pred_path: Path,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Evaluate a single sequence using py-motmetrics.

    Args:
        gt_path: Path to ground truth file (MOT format).
        pred_path: Path to tracker output file (MOT format).
        iou_threshold: IoU threshold for matching predictions to GT.

    Returns:
        Dictionary of evaluation metrics.
    """
    import motmetrics as mm

    gt_data = load_mot_file(gt_path)
    pred_data = load_mot_file(pred_path)

    # Create MOTAccumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Get all frame indices
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

    for frame_id in all_frames:
        gt_objs = gt_data.get(frame_id, [])
        pred_objs = pred_data.get(frame_id, [])

        # Extract IDs and boxes
        gt_ids = [obj["id"] for obj in gt_objs]
        pred_ids = [obj["id"] for obj in pred_objs]

        gt_boxes = np.array(
            [[obj["bb_left"], obj["bb_top"], obj["bb_width"], obj["bb_height"]]
             for obj in gt_objs]
        ) if gt_objs else np.empty((0, 4))

        pred_boxes = np.array(
            [[obj["bb_left"], obj["bb_top"], obj["bb_width"], obj["bb_height"]]
             for obj in pred_objs]
        ) if pred_objs else np.empty((0, 4))

        # Compute distance matrix (1 - IoU, so lower = better match)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
            dist_mat = 1 - iou_mat
            # Mask out pairs below IoU threshold
            dist_mat[dist_mat > (1 - iou_threshold)] = np.nan
        else:
            dist_mat = np.empty((len(gt_boxes), len(pred_boxes)))

        acc.update(gt_ids, pred_ids, dist_mat)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota", "motp", "idf1",
            "num_switches", "num_false_positives",
            "num_misses", "num_objects", "num_predictions",
            "mostly_tracked", "mostly_lost", "partially_tracked",
        ],
        name="sequence",
    )

    results = {
        "mota": float(summary["mota"].iloc[0]),
        "motp": float(summary["motp"].iloc[0]),
        "idf1": float(summary["idf1"].iloc[0]),
        "id_switches": int(summary["num_switches"].iloc[0]),
        "false_positives": int(summary["num_false_positives"].iloc[0]),
        "false_negatives": int(summary["num_misses"].iloc[0]),
        "num_objects": int(summary["num_objects"].iloc[0]),
        "num_predictions": int(summary["num_predictions"].iloc[0]),
        "mostly_tracked": int(summary["mostly_tracked"].iloc[0]),
        "mostly_lost": int(summary["mostly_lost"].iloc[0]),
        "partially_tracked": int(summary["partially_tracked"].iloc[0]),
    }

    return results


def evaluate_multiple_sequences(
    gt_dir: Path,
    pred_dir: Path,
    sequence_names: list[str],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Evaluate multiple sequences and compute aggregate metrics.

    Expects each sequence to have its own gt.txt and pred.txt file:
        gt_dir/{sequence_name}/gt/gt.txt
        pred_dir/{sequence_name}.txt

    Args:
        gt_dir: Root directory containing GT sequences.
        pred_dir: Directory containing per-sequence prediction files.
        sequence_names: List of sequence names to evaluate.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dictionary with per-sequence and aggregate metrics.
    """
    import motmetrics as mm

    all_accumulators = []
    per_sequence = {}

    for seq_name in sequence_names:
        gt_path = gt_dir / seq_name / "gt" / "gt.txt"
        pred_path = pred_dir / f"{seq_name}.txt"

        if not gt_path.is_file():
            logger.warning(f"GT not found for {seq_name}: {gt_path}")
            continue
        if not pred_path.is_file():
            logger.warning(f"Pred not found for {seq_name}: {pred_path}")
            continue

        try:
            result = evaluate_sequence(gt_path, pred_path, iou_threshold)
            per_sequence[seq_name] = result
            logger.info(
                f"{seq_name}: MOTA={result['mota']:.3f}, "
                f"IDF1={result['idf1']:.3f}, "
                f"IDSW={result['id_switches']}"
            )
        except Exception as e:
            logger.error(f"Error evaluating {seq_name}: {e}")

    # Compute aggregate metrics (macro average)
    if per_sequence:
        aggregate = {}
        metric_keys = ["mota", "motp", "idf1"]
        for key in metric_keys:
            values = [v[key] for v in per_sequence.values() if not np.isnan(v[key])]
            aggregate[key] = float(np.mean(values)) if values else 0.0

        int_keys = [
            "id_switches", "false_positives", "false_negatives",
            "num_objects", "num_predictions",
        ]
        for key in int_keys:
            aggregate[key] = sum(v[key] for v in per_sequence.values())
    else:
        aggregate = {}

    return {
        "per_sequence": per_sequence,
        "aggregate": aggregate,
    }


def print_metrics_table(metrics: dict) -> None:
    """
    Print a formatted table of evaluation metrics.

    Args:
        metrics: Dictionary from evaluate_sequence or aggregate results.
    """
    print("\n" + "=" * 60)
    print("         MOT Evaluation Results")
    print("=" * 60)

    if "aggregate" in metrics:
        agg = metrics["aggregate"]
        print(f"\n  {'Metric':<25} {'Value':>12}")
        print(f"  {'-' * 37}")
        print(f"  {'MOTA':<25} {agg.get('mota', 0)*100:>11.1f}%")
        print(f"  {'MOTP':<25} {agg.get('motp', 0)*100:>11.1f}%")
        print(f"  {'IDF1':<25} {agg.get('idf1', 0)*100:>11.1f}%")
        print(f"  {'ID Switches':<25} {agg.get('id_switches', 0):>12d}")
        print(f"  {'False Positives':<25} {agg.get('false_positives', 0):>12d}")
        print(f"  {'False Negatives':<25} {agg.get('false_negatives', 0):>12d}")
        print(f"  {'Total GT Objects':<25} {agg.get('num_objects', 0):>12d}")

        if "per_sequence" in metrics:
            print(f"\n  Per-sequence results ({len(metrics['per_sequence'])} sequences):")
            for name, seq_metrics in metrics["per_sequence"].items():
                print(
                    f"    {name:<30} "
                    f"MOTA={seq_metrics['mota']*100:5.1f}% "
                    f"IDF1={seq_metrics['idf1']*100:5.1f}% "
                    f"IDSW={seq_metrics['id_switches']:4d}"
                )
    else:
        print(f"\n  {'Metric':<25} {'Value':>12}")
        print(f"  {'-' * 37}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:<25} {value*100:>11.1f}%")
            else:
                print(f"  {key:<25} {value:>12}")

    print("=" * 60 + "\n")
