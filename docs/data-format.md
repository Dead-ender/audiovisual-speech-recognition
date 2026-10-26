# Data format

The project consumes a CSV manifest with one utterance per row:

| Column | Meaning |
| --- | --- |
| `audio_path` | Path to a WAV recording (relative to the CSV or absolute). |
| `video_path` | Path to a video file for the same utterance (relative to the CSV or absolute). |
| `transcript` | Human-verified transcript for that exact utterance. |

Example layout:

```text
corpus/
├── train.csv
├── validation.csv
├── audio/
│   ├── utterance-001.wav
│   └── utterance-002.wav
└── video/
    ├── utterance-001.mp4
    └── utterance-002.mp4
```

```csv
audio_path,video_path,transcript
audio/utterance-001.wav,video/utterance-001.mp4,hello world
audio/utterance-002.wav,video/utterance-002.mp4,turn left
```

## Alignment and preprocessing assumptions

- Each audio/video pair covers the same short utterance, starts at a comparable time, and has the matching transcript.
- Audio is decoded by TorchAudio, converted to mono, resampled to the configured sample rate, then transformed to a log-Mel spectrogram.
- Video is sampled uniformly to the configured frame count, converted to grayscale, and resized. Supply a stable crop centered on the speaker's mouth; face detection, lip localization, and crop tracking are not implemented.
- The model linearly resamples the video feature sequence to the audio feature sequence length. This assumes utterance-level alignment and does not model explicit audio/video offsets.
- The default character vocabulary supports lowercase English letters, spaces, and apostrophes. Normalize or transcribe out-of-vocabulary symbols before running `validate-manifest`.
- The training and validation manifests must be disjoint by speaker when the goal is speaker-independent evaluation. Do not split adjacent clips from the same recording across train and test.

`avsr validate-manifest corpus/train.csv` checks the required columns, file existence, non-empty transcripts, and supported characters. It does not decode media or audit the corpus license.

## Privacy and data rights

This repository contains no corpus. Speech and face recordings may identify participants. Confirm consent, access controls, retention requirements, and the source dataset's license before processing or publishing any data. Keep media, derived arrays, predictions, and checkpoints outside Git; the `.gitignore` excludes common data/artifact directories.
