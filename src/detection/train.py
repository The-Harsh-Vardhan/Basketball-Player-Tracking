"""
YOLO11 model training for basketball player detection.

Fine-tunes a pretrained YOLO11m model on the SportsMOT basketball subset
with Weights & Biases integration for experiment tracking.

Model choice justification (YOLO11m):
    - mAP@50:95 on COCO: 51.5 (vs 47.0 for YOLO11s, 39.5 for YOLO11n)
    - Inference on T4: ~4.7ms = 213 FPS (well above 30 FPS requirement)
    - Sweet spot: YOLO11n/s sacrifice too much accuracy for speed we don't need;
      YOLO11l/x offer diminishing mAP returns with higher compute cost.
    - 20.1M parameters — manageable for fine-tuning on a single GPU.
"""

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default training hyperparameters
DEFAULT_TRAIN_CONFIG = {
    "model": "yolo11m.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 10,        # Early stopping patience
    "optimizer": "AdamW",
    "lr0": 0.001,          # Initial learning rate
    "lrf": 0.01,           # Final learning rate (lr0 * lrf)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,        # Cosine annealing LR scheduler
    "mosaic": 1.0,         # Mosaic augmentation (helps with occlusion)
    "mixup": 0.15,         # MixUp augmentation
    "close_mosaic": 10,    # Disable mosaic for last N epochs
    "hsv_h": 0.015,        # HSV-Hue augmentation
    "hsv_s": 0.7,          # HSV-Saturation augmentation
    "hsv_v": 0.4,          # HSV-Value augmentation
    "degrees": 0.0,        # Rotation augmentation
    "translate": 0.1,      # Translation augmentation
    "scale": 0.5,          # Scale augmentation
    "fliplr": 0.5,         # Horizontal flip probability
    "flipud": 0.0,         # Vertical flip (disabled — basketball has gravity)
}


class YOLOTrainer:
    """
    Manages YOLO11 training for basketball player detection.

    Wraps the Ultralytics YOLO API with project-specific defaults,
    W&B integration, and configuration management.

    Attributes:
        model: Ultralytics YOLO model instance.
        data_config: Path to the dataset YAML configuration.
        project_dir: Directory for saving training runs.
    """

    def __init__(
        self,
        data_config: str | Path,
        model_name: str = "yolo11m.pt",
        project_dir: str | Path = "runs/detect",
        wandb_enabled: bool = True,
    ):
        """
        Initialize the trainer.

        Args:
            data_config: Path to the YOLO dataset.yaml.
            model_name: Pretrained model to load (e.g., 'yolo11m.pt').
            project_dir: Directory for saving training outputs.
            wandb_enabled: Whether to enable W&B logging.
        """
        from ultralytics import YOLO

        self.data_config = Path(data_config).resolve()
        self.project_dir = Path(project_dir)
        self.model_name = model_name
        self.wandb_enabled = wandb_enabled

        if not self.data_config.is_file():
            raise FileNotFoundError(
                f"Dataset config not found: {self.data_config}"
            )

        # Initialize W&B if enabled
        if wandb_enabled:
            self._setup_wandb()

        # Load pretrained model
        logger.info(f"Loading pretrained model: {model_name}")
        self.model = YOLO(model_name)

    def _setup_wandb(self) -> None:
        """Configure Weights & Biases integration."""
        try:
            import wandb

            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
                logger.info("W&B login successful")
            else:
                logger.warning(
                    "WANDB_API_KEY not set. W&B will prompt for login or run offline."
                )

            # Enable W&B in Ultralytics settings
            from ultralytics import settings
            settings.update({"wandb": True})

        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.wandb_enabled = False

    def train(
        self,
        run_name: str = "yolo11m_basketball",
        **kwargs,
    ) -> dict:
        """
        Train the YOLO model on the basketball dataset.

        Merges default hyperparameters with any user overrides.

        Args:
            run_name: Name for this training run (used in W&B and output dirs).
            **kwargs: Override any default training hyperparameter.

        Returns:
            Dictionary of training results (metrics).
        """
        # Merge defaults with overrides
        train_args = {**DEFAULT_TRAIN_CONFIG, **kwargs}

        # Remove 'model' from train args — it's passed separately
        train_args.pop("model", None)

        logger.info(f"Starting training run: {run_name}")
        logger.info(f"Dataset config: {self.data_config}")
        logger.info(f"Training args: {train_args}")

        results = self.model.train(
            data=str(self.data_config),
            project=str(self.project_dir),
            name=run_name,
            **train_args,
        )

        logger.info("Training complete")
        return results

    def validate(self, weights_path: str | Path | None = None) -> dict:
        """
        Run validation on the trained model.

        Args:
            weights_path: Path to model weights. If None, uses current model.

        Returns:
            Validation metrics dictionary.
        """
        from ultralytics import YOLO

        if weights_path:
            model = YOLO(str(weights_path))
        else:
            model = self.model

        logger.info("Running validation...")
        metrics = model.val(data=str(self.data_config))

        logger.info(
            f"Validation results — "
            f"mAP@50: {metrics.box.map50:.4f}, "
            f"mAP@50:95: {metrics.box.map:.4f}"
        )
        return metrics

    def export(
        self,
        weights_path: str | Path | None = None,
        format: str = "onnx",
    ) -> str:
        """
        Export the model to a specified format.

        Args:
            weights_path: Path to model weights. If None, uses current model.
            format: Export format ('onnx', 'torchscript', 'engine', etc.).

        Returns:
            Path to the exported model.
        """
        from ultralytics import YOLO

        if weights_path:
            model = YOLO(str(weights_path))
        else:
            model = self.model

        logger.info(f"Exporting model to {format}...")
        export_path = model.export(format=format)
        logger.info(f"Model exported to: {export_path}")
        return export_path


def load_training_config(config_path: Path) -> dict:
    """
    Load training configuration from a YAML file.

    The YAML file can override any default training hyperparameter.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of training configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}
