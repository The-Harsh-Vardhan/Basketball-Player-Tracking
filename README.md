# Basketball Player Tracking — YOLO11 + ByteTrack

A modular, Dockerized end-to-end **Multi-Object Tracking (MOT)** pipeline for tracking basketball players in broadcast video. Combines **YOLO11m** detection with **ByteTrack** (basketball-tuned Kalman Filter) to maintain persistent player IDs through occlusions, screens, and camera motion.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline (>30 FPS on T4)            │
│                                                                  │
│  ┌──────────┐    ┌────────────┐    ┌───────────┐    ┌────────┐  │
│  │  Video /  │───▶│  YOLO11m   │───▶│ ByteTrack │───▶│  MOT   │  │
│  │  Frames   │    │ Detection  │    │ Tracking  │    │ Output │  │
│  └──────────┘    └────────────┘    └───────────┘    └────────┘  │
│                    conf: 0.25         │                           │
│                    imgsz: 640         │                           │
│                    FP16: ✓            ▼                           │
│                                ┌─────────────┐                   │
│                                │ Basketball  │                   │
│                                │ Kalman      │                   │
│                                │ Filter      │                   │
│                                │ (tuned)     │                   │
│                                └─────────────┘                   │
└─────────────────────────────────────────────────────────────────┘

Output:
  ├── tracker_output.txt  (MOT format)
  ├── output.mp4          (annotated video)
  └── MOTA / IDF1 metrics
```

## Project Structure

```
├── configs/
│   ├── dataset.yaml              # YOLO dataset config
│   └── tracker.yaml              # ByteTrack hyperparameters
├── src/
│   ├── data/
│   │   ├── download.py           # Dataset validation & instructions
│   │   ├── filter_basketball.py  # SportsMOT basketball subset isolation
│   │   └── parse_mot.py          # gt.txt → YOLO annotation converter
│   ├── detection/
│   │   └── train.py              # YOLO11 training with W&B
│   ├── tracking/
│   │   ├── bytetrack.py          # ByteTrack wrapper
│   │   └── kalman_filter.py      # Basketball-tuned KalmanFilterXYAH
│   ├── evaluation/
│   │   └── metrics.py            # MOTA, IDF1, MOTP computation
│   └── inference/
│       ├── pipeline.py           # End-to-end detect → track → output
│       └── visualize.py          # Video rendering with annotations
├── scripts/
│   ├── prepare_data.py           # Data preparation orchestrator
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   └── run_inference.py          # Inference entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── tests/
```

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support ([nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- SportsMOT dataset downloaded to `data/SportsMOT/` (see [Dataset Setup](#dataset-setup))

### Docker (Recommended)

```bash
# Build the container
docker-compose build

# Step 1: Prepare dataset
docker-compose run data-prep

# Step 2: Train detector
docker-compose run trainer

# Step 3: Run inference
docker-compose run tracker --source /app/data/video.mp4

# Step 4: Evaluate
docker-compose run evaluator --gt-dir /app/data/SportsMOT/dataset/val --pred-dir /app/outputs/predictions
```

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare dataset
python scripts/prepare_data.py --dataset-root data/SportsMOT --output-dir data/basketball_yolo

# Step 2: Train YOLO11m
python scripts/train.py --config configs/dataset.yaml --epochs 100

# Step 3: Run inference
python scripts/run_inference.py --source data/SportsMOT/dataset/val/<sequence>/img1/

# Step 4: Evaluate
python scripts/evaluate.py --gt-file data/SportsMOT/dataset/val/<seq>/gt/gt.txt --pred-file outputs/tracker_output.txt
```

## Dataset Setup

This pipeline uses the **SportsMOT** dataset (ICCV 2023), specifically the basketball subset.

1. Download from [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/12424)
2. Extract to `data/SportsMOT/`
3. Run `python scripts/prepare_data.py` to:
   - Filter basketball sequences using `splits_txt/basketball.txt`
   - Convert MOT-format `gt.txt` → normalized YOLO annotations
   - Organize into `data/basketball_yolo/{images,labels}/{train,val}/`

Expected structure:
```
data/SportsMOT/
├── splits_txt/
│   ├── basketball.txt    # Basketball sequence names
│   ├── train.txt         # Training split
│   └── val.txt           # Validation split
└── dataset/
    ├── train/<seq>/gt/gt.txt, img1/, seqinfo.ini
    └── val/<seq>/gt/gt.txt, img1/, seqinfo.ini
```

## Architectural Decisions

### 1. Detection Model: YOLO11m

| Model | mAP@50:95 | Inference (T4) | FPS (T4) | Decision |
|-------|-----------|---------------|----------|----------|
| YOLO11n | 39.5 | 1.5ms | 667 | Too low mAP |
| YOLO11s | 47.0 | 2.5ms | 400 | Acceptable |
| **YOLO11m** | **51.5** | **4.7ms** | **213** | **Selected** ✓ |
| YOLO11l | 53.4 | 6.2ms | 161 | Diminishing returns |
| YOLO11x | 54.7 | 11.3ms | 88 | Overkill for single-class |

**Rationale**: YOLO11m offers the best mAP-to-speed tradeoff. At 213 FPS detection + ~1ms ByteTrack overhead, the pipeline operates well above the 30 FPS requirement. The 51.5 mAP provides robust detection under occlusion and varied lighting.

### 2. Tracking Engine: ByteTrack (via boxmot)

ByteTrack's two-stage association is specifically advantageous for basketball:

- **Stage 1** (high-confidence): IoU-based matching of confident detections to existing tracks
- **Stage 2** (low-confidence): Matches remaining tracks against low-confidence detections — this rescues partially occluded players during screens

We use the `boxmot` library (v16+) for a battle-tested ByteTrack implementation, focusing our engineering effort on Kalman Filter tuning rather than algorithm reimplementation.

### 3. Kalman Filter Tuning for Basketball Dynamics

| Parameter | Default | Basketball-Tuned | Rationale |
|-----------|---------|-----------------|-----------|
| `_std_weight_position` | 1/20 (0.050) | **1/15 (0.067)** | Basketball players move 3-8 m/s. Larger position uncertainty prevents the filter from "locking" during fast crossovers. |
| `_std_weight_velocity` | 1/160 (0.00625) | **1/100 (0.010)** | High jerk: players accelerate/decelerate rapidly (cuts, screens, sprints). Larger velocity noise allows faster adaptation. |

The noise is proportional to bounding box height — a built-in scaling proxy for camera distance. Farther players have smaller boxes and proportionally smaller pixel-space motion.

### 4. ByteTrack Hyperparameters

```yaml
track_thresh: 0.5   # Screen detections score 0.1-0.5, captured by Stage 2
match_thresh: 0.8   # Liberal IoU match — basketball players are well-separated
track_buffer: 75    # 3s at 25 FPS — covers boundary re-entry window
frame_rate: 25      # SportsMOT native frame rate
min_hits: 3         # Require 3 consecutive detections to confirm a track
```

## Edge Case Analysis

### 1. The "Screen" Test (5-10 frame occlusion)

**Scenario**: Player A sets a screen, completely obscuring Player B for 5-10 frames.

**How ByteTrack handles it**:
1. When the occluded player's detection confidence drops (partial occlusion), Stage 2 matching catches these low-confidence detections and associates them with the existing track via Kalman-predicted position.
2. During complete occlusion (0 frames of detection), the track enters "lost" state. The Kalman filter continues predicting position using the learned velocity.
3. When the player emerges, the predicted position is close enough for IoU matching to succeed, preserving the original track ID.
4. Our tuned KF (higher velocity noise) keeps predictions accurate during the acceleration changes that often accompany screens.

**Expected behavior**: ID is preserved through 5-10 frame occlusions in most cases. The `track_buffer=75` ensures lost tracks survive far longer than needed.

### 2. Boundary Exit (3-second re-entry)

**Scenario**: A player exits the camera frame and re-enters 3 seconds later (~75 frames at 25 FPS).

**How our system handles it**:
- `track_buffer=75` keeps the lost track alive for the full 3-second window
- The Kalman filter continues predicting position (will drift significantly over 75 frames)
- **Re-identification depends on re-entry location**:
  - If the player re-enters **near their exit point** (e.g., same side of court), IoU-based matching can re-associate the track. **ID is preserved**.
  - If the player re-enters **at a distant location** (e.g., ran across behind the camera), the predicted position will be too far for IoU matching. **A new ID is assigned**.

**Documented assumption**: This is a fundamental limitation of motion-only tracking (ByteTrack). Without appearance-based re-identification (ReID), there is no mechanism to match players across large spatial gaps. For production use, we recommend BoTSORT with a lightweight ReID model (e.g., OSNet) — boxmot supports this with a one-line configuration change.

## W&B Dashboard

> **Link**: `[TODO: Add public W&B dashboard link after training]`
>
> The W&B dashboard includes:
> - Training loss curves (box loss, cls loss, dfl loss)
> - Validation mAP@50 and mAP@50:95 over epochs
> - Hyperparameter comparison (if sweep was run)
> - Model artifact versioning
> - Sample detection visualizations

To enable W&B:
```bash
export WANDB_API_KEY=your_key_here
# Or copy .env.example to .env and fill in your key
```

## Evaluation Results

> **[TODO: Fill after training and evaluation]**

| Metric | Value | Target |
|--------|-------|--------|
| MOTA | TBD | > 70% |
| IDF1 | TBD | > 65% |
| MOTP | TBD | — |
| ID Switches | TBD | Low |
| FPS (T4) | TBD | > 30 |

Run evaluation:
```bash
python scripts/evaluate.py \
    --gt-dir data/SportsMOT/dataset/val \
    --pred-dir outputs/predictions \
    --output-json outputs/eval_results.json
```

## Environment Reproduction

### Docker (single command)

```bash
docker-compose build
docker-compose run tracker --source /app/data/video.mp4
```

### Manual

```bash
pip install -r requirements.txt
# Requires: CUDA 12.1+, Python 3.11+
```

## License

This project is for assessment purposes. SportsMOT dataset is subject to its [original license](https://github.com/MCG-NJU/SportsMOT).
