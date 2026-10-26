"""End-to-end smoke test for generated WAV/video through decoding."""

import csv
import wave

import numpy as np
import pytest
import torch

from avsr.config import AVSRConfig
from avsr.data import ManifestDataset, collate_audio_visual
from avsr.decode import greedy_ctc_decode
from avsr.model import AVSRModel


def _write_toy_wav(path):
    sample_rate = 16_000
    samples = (np.zeros(sample_rate // 2, dtype=np.int16)).tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples)


def test_generated_media_runs_through_dataset_model_and_decoder(tmp_path):
    torchaudio = pytest.importorskip("torchaudio")
    cv2 = pytest.importorskip("cv2")
    del torchaudio  # Import here to make missing optional local runtimes skip cleanly.

    audio_path = tmp_path / "toy.wav"
    video_path = tmp_path / "toy.avi"
    _write_toy_wav(audio_path)

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV has no MJPG video writer in this environment")
    try:
        for value in (32, 96, 160, 224):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    finally:
        writer.release()

    manifest = tmp_path / "smoke.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        rows = csv.writer(stream)
        rows.writerow(("audio_path", "video_path", "transcript"))
        rows.writerow((audio_path.name, video_path.name, "hi"))

    config = AVSRConfig(
        n_mels=16,
        n_frames=4,
        frame_height=16,
        frame_width=16,
        model_dim=32,
        num_heads=4,
        num_transformer_layers=1,
        dropout=0.0,
    )
    batch = collate_audio_visual([ManifestDataset(manifest, config)[0]])
    model = AVSRModel(config).eval()
    with torch.inference_mode():
        logits, lengths = model(
            batch["audio"],
            batch["video"],
            batch["audio_lengths"],
            batch["video_lengths"],
        )

    decoded = greedy_ctc_decode(logits, lengths)
    assert logits.shape[1] == 1
    assert lengths.tolist() == [batch["audio"].shape[-1]]
    assert len(decoded) == 1
    assert set(decoded[0]) <= set(config.charset)
