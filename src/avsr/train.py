"""Training and evaluation loops."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from avsr.checkpoint import save_checkpoint
from avsr.config import AVSRConfig
from avsr.data import ManifestDataset, collate_audio_visual
from avsr.decode import greedy_ctc_decode
from avsr.metrics import error_rates
from avsr.model import AVSRModel
from avsr.text import TextCodec


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _validate_ctc_lengths(batch: dict[str, Any], input_lengths: torch.Tensor) -> None:
    """Fail early when a transcript cannot fit the model's CTC time axis."""
    offset = 0
    flat_targets = batch["targets"].detach().cpu()
    for index, target_length in enumerate(batch["target_lengths"].tolist()):
        target = flat_targets[offset : offset + target_length]
        repeated = int((target[1:] == target[:-1]).sum()) if target_length > 1 else 0
        required = target_length + repeated
        available = int(input_lengths[index].item())
        if required > available:
            raise ValueError(
                f"example {index} needs at least {required} CTC frames for its transcript, "
                f"but the audio provides {available}; shorten or segment the utterance"
            )
        offset += target_length


def train_one_epoch(
    model: AVSRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    *,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    loss_total = 0.0
    examples = 0
    for original_batch in loader:
        batch = _move_batch(original_batch, device)
        logits, input_lengths = model(
            batch["audio"],
            batch["video"],
            batch["audio_lengths"],
            batch["video_lengths"],
        )
        _validate_ctc_lengths(batch, input_lengths)
        loss = criterion(
            F.log_softmax(logits, dim=-1),
            batch["targets"],
            input_lengths.cpu(),
            batch["target_lengths"].cpu(),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("CTC loss became non-finite; check input/target lengths")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        size = batch["audio"].shape[0]
        loss_total += float(loss.detach()) * size
        examples += size
    if examples == 0:
        raise ValueError("training manifest produced no batches")
    return loss_total / examples


@torch.inference_mode()
def evaluate(
    model: AVSRModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    codec = TextCodec(model.config.charset)
    loss_total = 0.0
    examples = 0
    references: list[str] = []
    hypotheses: list[str] = []
    for original_batch in loader:
        batch = _move_batch(original_batch, device)
        logits, input_lengths = model(
            batch["audio"],
            batch["video"],
            batch["audio_lengths"],
            batch["video_lengths"],
        )
        _validate_ctc_lengths(batch, input_lengths)
        loss = criterion(
            F.log_softmax(logits, dim=-1),
            batch["targets"],
            input_lengths.cpu(),
            batch["target_lengths"].cpu(),
        )
        size = batch["audio"].shape[0]
        loss_total += float(loss) * size
        examples += size
        hypotheses.extend(greedy_ctc_decode(logits, input_lengths, codec))
        references.extend(batch["transcripts"])
    if examples == 0:
        raise ValueError("validation manifest produced no batches")
    return {
        "loss": loss_total / examples,
        **error_rates(references, hypotheses),
        "examples": examples,
        "references": references,
        "hypotheses": hypotheses,
    }


def train_model(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output: str | Path,
    *,
    config: AVSRConfig | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    num_workers: int = 0,
    device: str | None = None,
) -> list[dict[str, float]]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    config = config or AVSRConfig()
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_data = ManifestDataset(train_manifest, config)
    validation_data = ManifestDataset(validation_manifest, config)
    loader_options = {"batch_size": batch_size, "collate_fn": collate_audio_visual}
    train_loader = DataLoader(train_data, shuffle=True, num_workers=num_workers, **loader_options)
    validation_loader = DataLoader(
        validation_data, shuffle=False, num_workers=num_workers, **loader_options
    )

    model = AVSRModel(config).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=config.blank_id, zero_infinity=True)
    history: list[dict[str, float]] = []
    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        training_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device_obj
        )
        validation = evaluate(model, validation_loader, criterion, device_obj)
        row = {
            "epoch": float(epoch),
            "train_loss": training_loss,
            "validation_loss": validation["loss"],
            "validation_cer": validation["cer"],
            "validation_wer": validation["wer"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={training_loss:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_cer={validation['cer']:.4f} val_wer={validation['wer']:.4f}"
        )
        if validation["cer"] < best_cer:
            best_cer = validation["cer"]
            save_checkpoint(
                model,
                output,
                epoch=epoch,
                optimizer=optimizer,
                validation_cer=best_cer,
            )

    history_path = Path(output).with_suffix(".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return history
