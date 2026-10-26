"""CSV manifests and paired audio/video loading."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from avsr.config import AVSRConfig
from avsr.text import TextCodec

REQUIRED_COLUMNS = ("audio_path", "video_path", "transcript")


@dataclass(frozen=True)
class ManifestRecord:
    audio_path: Path
    video_path: Path
    transcript: str


def read_manifest(path: str | Path, *, check_files: bool = True) -> list[ManifestRecord]:
    """Read a UTF-8 CSV manifest; relative media paths use the CSV's directory."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("manifest is empty; expected a CSV header")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            audio_value = (row.get("audio_path") or "").strip()
            video_value = (row.get("video_path") or "").strip()
            transcript = (row.get("transcript") or "").strip()
            if not audio_value or not video_value or not transcript:
                raise ValueError(f"line {line_number}: paths and transcript must be non-empty")
            audio_path = _resolve_media_path(audio_value, manifest_path.parent)
            video_path = _resolve_media_path(video_value, manifest_path.parent)
            if check_files:
                for label, media_path in (("audio", audio_path), ("video", video_path)):
                    if not media_path.is_file():
                        raise FileNotFoundError(
                            f"line {line_number}: {label} file not found: {media_path}"
                        )
            records.append(ManifestRecord(audio_path, video_path, transcript))

    if not records:
        raise ValueError("manifest contains no examples")
    return records


def _resolve_media_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def validate_manifest(path: str | Path, config: AVSRConfig | None = None) -> int:
    """Validate manifest structure, media files, and transcript vocabulary."""
    codec = TextCodec((config or AVSRConfig()).charset)
    records = read_manifest(path)
    for line_number, record in enumerate(records, start=2):
        try:
            codec.encode(record.transcript)
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
    return len(records)


class ManifestDataset(Dataset[dict[str, Any]]):
    """Load log-Mel audio, grayscale video frames, and text from a manifest."""

    def __init__(
        self,
        manifest: str | Path,
        config: AVSRConfig | None = None,
    ) -> None:
        self.config = config or AVSRConfig()
        self.records = read_manifest(manifest)
        self.codec = TextCodec(self.config.charset)
        try:
            import torchaudio
        except ImportError as error:
            raise RuntimeError(
                "Audio loading requires TorchAudio; install a build matching PyTorch."
            ) from error
        self._torchaudio = torchaudio
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.win_length,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform, sample_rate = self._torchaudio.load(str(record.audio_path))
        if waveform.ndim != 2 or waveform.shape[1] == 0:
            raise ValueError(f"audio file contains no samples: {record.audio_path}")
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            waveform = self._torchaudio.functional.resample(
                waveform, sample_rate, self.config.sample_rate
            )
        audio = self._to_db(self._mel(waveform)).squeeze(0)
        video = _load_video(
            record.video_path,
            num_frames=self.config.n_frames,
            height=self.config.frame_height,
            width=self.config.frame_width,
        )
        return {
            "audio": audio,
            "video": video,
            "targets": torch.tensor(self.codec.encode(record.transcript), dtype=torch.long),
            "transcript": self.codec.normalize(record.transcript),
        }


def _load_video(path: Path, *, num_frames: int, height: int, width: int) -> torch.Tensor:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Video loading requires OpenCV (opencv-python-headless).") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frames = _read_unknown_length_video(capture, cv2, height, width)
            if not frames:
                raise ValueError(f"video contains no readable frames: {path}")
            indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
            sampled = [frames[index] for index in indices]
        else:
            indices = np.linspace(0, frame_count - 1, num_frames).round().astype(int)
            sampled = []
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    if not sampled:
                        raise ValueError(f"could not read frame {frame_index} from {path}")
                    sampled.append(sampled[-1].copy())
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
                sampled.append(gray.astype(np.float32) / 255.0)
    finally:
        capture.release()
    return torch.from_numpy(np.stack(sampled)).float()


def _read_unknown_length_video(capture: Any, cv2: Any, height: int, width: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while len(frames) < 4096:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(gray.astype(np.float32) / 255.0)
    return frames


def collate_audio_visual(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length features and concatenate CTC targets."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_audio = max(item["audio"].shape[-1] for item in batch)
    max_video = max(item["video"].shape[0] for item in batch)
    audio_batch = []
    video_batch = []
    targets = []
    audio_lengths = []
    video_lengths = []
    target_lengths = []

    for item in batch:
        audio, video, target = item["audio"], item["video"], item["targets"]
        audio_lengths.append(audio.shape[-1])
        video_lengths.append(video.shape[0])
        target_lengths.append(target.numel())
        audio_batch.append(F.pad(audio, (0, max_audio - audio.shape[-1])))
        video_batch.append(F.pad(video, (0, 0, 0, 0, 0, max_video - video.shape[0])))
        targets.append(target)

    return {
        "audio": torch.stack(audio_batch),
        "video": torch.stack(video_batch),
        "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
        "video_lengths": torch.tensor(video_lengths, dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "transcripts": [item["transcript"] for item in batch],
    }
