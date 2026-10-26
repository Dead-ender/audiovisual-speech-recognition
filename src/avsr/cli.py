"""Command-line interface for manifest validation, training, and evaluation."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from avsr.checkpoint import load_checkpoint
from avsr.data import ManifestDataset, collate_audio_visual, validate_manifest
from avsr.decode import greedy_ctc_decode
from avsr.train import evaluate, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avsr",
        description="Audiovisual speech-recognition research toolkit (character CTC).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-manifest", help="check a CSV and its media paths")
    validate.add_argument("manifest", type=Path)

    train = commands.add_parser("train", help="train from paired audio/video CSV manifests")
    train.add_argument("--train-manifest", required=True, type=Path)
    train.add_argument("--validation-manifest", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--device", choices=("cpu", "cuda"), default=None)

    for name, description in (
        ("evaluate", "compute loss, CER, and WER for a labeled manifest"),
        ("predict", "decode a manifest and write JSON predictions"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--num-workers", type=int, default=0)
        command.add_argument("--device", choices=("cpu", "cuda"), default=None)
        if name == "predict":
            command.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-manifest":
        count = validate_manifest(args.manifest)
        print(f"Valid manifest: {count} examples")
        return
    if args.command == "train":
        train_model(
            args.train_manifest,
            args.validation_manifest,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            device=args.device,
        )
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, _ = load_checkpoint(args.checkpoint, device=device)
    dataset = ManifestDataset(args.manifest, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_visual,
    )
    if args.command == "evaluate":
        result = evaluate(model, loader, nn.CTCLoss(blank=config.blank_id, zero_infinity=True), device)
        result.pop("references")
        result.pop("hypotheses")
        print(json.dumps(result, indent=2))
        return

    predictions = []
    for batch in loader:
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        logits, lengths = model(
            audio,
            video,
            batch["audio_lengths"].to(device),
            batch["video_lengths"].to(device),
        )
        decoded = greedy_ctc_decode(logits, lengths)
        predictions.extend(
            {"reference": reference, "prediction": prediction}
            for reference, prediction in zip(batch["transcripts"], decoded)
        )
    serialized = json.dumps(predictions, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
