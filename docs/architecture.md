# Architecture and current scope

The implementation is a compact research baseline, not a claim of state-of-the-art lip reading.

1. `ManifestDataset` decodes WAV audio, converts it to mono, resamples it, and computes a log-Mel spectrogram. It samples a fixed number of grayscale frames from the supplied mouth-region video.
2. `AudioEncoder` applies a 2D CNN over mel bins and time while preserving the time axis. `VideoEncoder` applies a 3D CNN to frame sequences and spatially pools each time step.
3. Video embeddings are linearly interpolated to the audio time length. The concatenated features pass through a projection, sinusoidal positions, and a Transformer encoder. Padding masks prevent padded audio positions from serving as keys.
4. A character classifier emits frame-wise logits. PyTorch CTC loss trains the sequence alignment without frame-level character labels; greedy CTC decoding collapses repeats and blanks.

## Deliberate boundaries

- Input video is expected to be mouth-cropped and temporally aligned with the utterance. Face detection, landmark tracking, frame/audio synchronization, and robust media augmentation are future work.
- The default model uses full self-attention and should be used with short utterances; its attention memory grows quadratically with the number of audio frames.
- The output alphabet is character-level and English-only by default. Change the `charset` in `AVSRConfig` consistently for other languages.
- Evaluation reports corpus-level CER and WER using greedy decoding. There is no beam search, language model, speaker-independent benchmark, or pretrained checkpoint in this repository.
- The original notebook's synthetic task remains a wiring example only. Its random inputs and random labels have no semantic relationship.

## Checkpoints

Checkpoints contain a PyTorch state dictionary, model configuration, epoch number, optional optimizer state, and the validation CER used to select the checkpoint. Checkpoints are written atomically. Do not load weights from sources you do not trust.
