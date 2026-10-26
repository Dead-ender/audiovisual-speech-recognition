import csv

import pytest
import torch
from torch import nn

from avsr.checkpoint import load_checkpoint, save_checkpoint
from avsr.cli import build_parser
from avsr.config import AVSRConfig
from avsr.data import collate_audio_visual, validate_manifest
from avsr.decode import greedy_ctc_decode
from avsr.metrics import edit_distance, error_rates
from avsr.model import AVSRModel
from avsr.text import TextCodec
from avsr.train import _validate_ctc_lengths


def small_config() -> AVSRConfig:
    return AVSRConfig(
        n_mels=16,
        n_frames=4,
        frame_height=16,
        frame_width=16,
        model_dim=32,
        num_heads=4,
        num_transformer_layers=1,
        dropout=0.0,
    )


def test_text_codec_normalizes_and_round_trips():
    codec = TextCodec()
    encoded = codec.encode("  Hello   world  ")
    assert codec.decode(encoded) == "hello world"
    assert 0 not in encoded


def test_text_codec_rejects_characters_outside_vocabulary():
    with pytest.raises(ValueError, match="unsupported characters"):
        TextCodec().encode("hello!")


def test_config_rejects_invalid_attention_dimensions():
    with pytest.raises(ValueError, match="num_heads must be positive"):
        AVSRConfig(num_heads=0)


def test_collate_pads_features_and_tracks_true_lengths():
    batch = collate_audio_visual(
        [
            {
                "audio": torch.ones(16, 12),
                "video": torch.ones(4, 16, 16),
                "targets": torch.tensor([8, 9]),
                "transcript": "hi",
            },
            {
                "audio": torch.ones(16, 8),
                "video": torch.ones(3, 16, 16),
                "targets": torch.tensor([8]),
                "transcript": "h",
            },
        ]
    )
    assert batch["audio"].shape == (2, 16, 12)
    assert batch["video"].shape == (2, 4, 16, 16)
    assert batch["audio_lengths"].tolist() == [12, 8]
    assert batch["video_lengths"].tolist() == [4, 3]
    assert batch["target_lengths"].tolist() == [2, 1]
    assert batch["targets"].tolist() == [8, 9, 8]


def test_model_forward_ctc_loss_and_backward():
    config = small_config()
    model = AVSRModel(config)
    audio = torch.randn(2, config.n_mels, 24)
    video = torch.rand(2, config.n_frames, config.frame_height, config.frame_width)
    logits, lengths = model(
        audio,
        video,
        audio_lengths=torch.tensor([20, 16]),
        video_lengths=torch.tensor([4, 3]),
    )
    codec = TextCodec(config.charset)
    targets = torch.tensor(codec.encode("hello") + codec.encode("hi"), dtype=torch.long)
    loss = nn.CTCLoss(blank=config.blank_id, zero_infinity=True)(
        logits.log_softmax(-1),
        targets,
        lengths.cpu(),
        torch.tensor([5, 2]),
    )
    assert logits.shape == (24, 2, config.vocab_size)
    assert lengths.tolist() == [20, 16]
    assert torch.isfinite(loss)
    loss.backward()


def test_ctc_length_validation_counts_repeated_labels():
    batch = {"targets": torch.tensor([4, 4, 5]), "target_lengths": torch.tensor([3])}
    with pytest.raises(ValueError, match="at least 4 CTC frames"):
        _validate_ctc_lengths(batch, torch.tensor([3]))
    _validate_ctc_lengths(batch, torch.tensor([4]))


def test_greedy_ctc_collapses_repeats_and_blanks():
    codec = TextCodec()
    hi, blank = codec.char_to_id["h"], codec.blank_id
    labels = torch.tensor([[hi], [hi], [blank], [hi], [codec.char_to_id["i"]]])
    logits = torch.full((5, 1, codec.char_to_id["i"] + 1), -10.0)
    logits.scatter_(2, labels.unsqueeze(-1), 10.0)
    assert greedy_ctc_decode(logits, torch.tensor([5]), codec) == ["hhi"]


def test_cer_and_wer_are_corpus_level():
    assert edit_distance("kitten", "sitting") == 3
    rates = error_rates(["turn left now"], ["turn right now"])
    assert rates["wer"] == pytest.approx(1 / 3)
    assert 0 < rates["cer"] < 1


def test_manifest_validation_resolves_relative_paths(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "video").mkdir()
    (tmp_path / "audio" / "one.wav").write_bytes(b"placeholder")
    (tmp_path / "video" / "one.mp4").write_bytes(b"placeholder")
    manifest = tmp_path / "train.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("audio_path", "video_path", "transcript"))
        writer.writeheader()
        writer.writerow(
            {"audio_path": "audio/one.wav", "video_path": "video/one.mp4", "transcript": "Hello"}
        )
    assert validate_manifest(manifest, small_config()) == 1


def test_manifest_validation_reports_missing_media(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text(
        "audio_path,video_path,transcript\nmissing.wav,missing.mp4,hello\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        validate_manifest(manifest)


def test_checkpoint_round_trip_preserves_config_and_weights(tmp_path):
    config = small_config()
    model = AVSRModel(config)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, checkpoint_path, epoch=3, validation_cer=0.4)
    restored, restored_config, metadata = load_checkpoint(checkpoint_path)
    assert restored_config == config
    assert metadata["epoch"] == 3
    assert metadata["validation_cer"] == pytest.approx(0.4)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])


def test_cli_parser_exposes_data_and_model_commands():
    args = build_parser().parse_args(
        ["train", "--train-manifest", "train.csv", "--validation-manifest", "val.csv", "--output", "model.pt"]
    )
    assert args.command == "train"
    assert args.epochs == 10
