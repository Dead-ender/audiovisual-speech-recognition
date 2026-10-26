"""Audio, video, and Transformer-fusion model with CTC output."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from avsr.config import AVSRConfig


class AudioEncoder(nn.Module):
    """2D CNN over a log-Mel spectrogram; preserves the time axis."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.project = nn.Conv1d(128, output_dim, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        encoded = self.convs(mel.unsqueeze(1))
        encoded = encoded.mean(dim=2)
        return self.project(encoded).transpose(1, 2)


class VideoEncoder(nn.Module):
    """3D CNN over grayscale video; spatially pools and preserves time."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        self.project = nn.Linear(64, output_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        encoded = self.convs(video.unsqueeze(1)).mean(dim=(-1, -2))
        return self.project(encoded.transpose(1, 2))


class AVSRModel(nn.Module):
    """Fuse audio/video features and emit time-major CTC logits.

    Args:
        config: Feature, vocabulary, and Transformer dimensions.

    Returns:
        A pair ``(logits, output_lengths)`` where logits have shape ``(T, B, V)``.
    """

    def __init__(self, config: AVSRConfig | None = None) -> None:
        super().__init__()
        self.config = config or AVSRConfig()
        self.audio_encoder = AudioEncoder(self.config.model_dim)
        self.video_encoder = VideoEncoder(self.config.model_dim)
        self.fusion = nn.Linear(self.config.model_dim * 2, self.config.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.model_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=self.config.num_transformer_layers
        )
        self.norm = nn.LayerNorm(self.config.model_dim)
        self.classifier = nn.Linear(self.config.model_dim, self.config.vocab_size)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        video_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, padded_audio_length = audio.shape
        if audio_lengths is None:
            audio_lengths = torch.full(
                (batch_size,), padded_audio_length, dtype=torch.long, device=audio.device
            )
        if video_lengths is None:
            video_lengths = torch.full(
                (batch_size,), video.shape[1], dtype=torch.long, device=video.device
            )
        audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
        video_lengths = video_lengths.to(device=video.device, dtype=torch.long)
        if torch.any(audio_lengths <= 0) or torch.any(audio_lengths > padded_audio_length):
            raise ValueError("audio_lengths must be within the padded audio time dimension")
        if torch.any(video_lengths <= 0) or torch.any(video_lengths > video.shape[1]):
            raise ValueError("video_lengths must be within the padded video time dimension")

        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)
        aligned_video = []
        for index in range(batch_size):
            audio_length = int(audio_lengths[index].item())
            video_length = int(video_lengths[index].item())
            sequence = video_features[index, :video_length].transpose(0, 1).unsqueeze(0)
            sequence = F.interpolate(
                sequence, size=audio_length, mode="linear", align_corners=False
            ).squeeze(0).transpose(0, 1)
            aligned_video.append(sequence)
        video_features = pad_sequence(aligned_video, batch_first=True)
        if video_features.shape[1] < padded_audio_length:
            video_features = F.pad(
                video_features, (0, 0, 0, padded_audio_length - video_features.shape[1])
            )

        fused = self.fusion(torch.cat((audio_features, video_features), dim=-1))
        fused = fused + _sinusoidal_positions(
            fused.shape[1], fused.shape[2], fused.device, fused.dtype
        )
        positions = torch.arange(fused.shape[1], device=fused.device).unsqueeze(0)
        padding_mask = positions >= audio_lengths.unsqueeze(1)
        encoded = self.transformer(fused, src_key_padding_mask=padding_mask)
        logits = self.classifier(self.norm(encoded)).transpose(0, 1)
        return logits, audio_lengths


def _sinusoidal_positions(
    length: int, dimension: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros((length, dimension), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * scales)
    encoding[:, 1::2] = torch.cos(positions * scales)
    return encoding.to(dtype=dtype).unsqueeze(0)
