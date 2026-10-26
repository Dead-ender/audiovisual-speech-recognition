"""Safe, atomic checkpoint save/load helpers."""

from pathlib import Path

import torch

from avsr.config import AVSRConfig
from avsr.model import AVSRModel


def save_checkpoint(
    model: AVSRModel,
    path: str | Path,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    validation_cer: float | None = None,
) -> None:
    """Write a state-dict checkpoint, replacing the destination atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "epoch": epoch,
        "validation_cer": validation_cer,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[AVSRModel, AVSRConfig, dict[str, object]]:
    """Load model weights and their preprocessing configuration."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = AVSRConfig(**checkpoint["config"])
    model = AVSRModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint
