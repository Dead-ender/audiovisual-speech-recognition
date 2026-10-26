# Model Card: AVSR Fusion Baseline

## Summary

This repository provides an audiovisual speech-recognition research baseline. It encodes log-Mel audio and grayscale mouth-region video with separate convolutional networks, aligns and fuses the features with a Transformer encoder, and predicts character sequences with CTC.

## Model status

- **Model type:** untrained architecture and training/inference code.
- **Checkpoint included:** no.
- **Training dataset:** none included; no training run is claimed by this repository.
- **Evaluation data or results:** none included; no CER, WER, or real-world accuracy claim is made.
- **Intended use:** educational experimentation and research prototyping with properly licensed, consented data.

## Inputs and outputs

The data loader expects a mono or multichannel WAV file, a synchronized video clip already cropped to the mouth region, and a transcript using the configured character vocabulary. Audio is converted to log-Mel features; video is sampled at a fixed number of grayscale frames. The model emits character logits and uses CTC decoding to produce text.

## Limitations

- There are no face or lip detectors, alignment tools, or production input checks.
- The default character vocabulary supports lowercase English letters, spaces, and apostrophes.
- The baseline has not been evaluated for speakers, languages, environments, demographics, or recording devices.
- Performance and failure modes are unknown until a reproducible dataset split and evaluation protocol are used.
- The notebook's random synthetic tensors are a model-wiring illustration, not speech data or evidence of recognition quality.

## Data and privacy

No participant data or pretrained weights are distributed here. Audio and face video may contain identifying biometric information. Obtain consent, follow dataset terms and applicable privacy requirements, and avoid publishing private recordings or derived artifacts.

## Reproducibility

Install the dependencies described in the README, validate manifests before training, record the dataset version and speaker split, and preserve the training command, configuration, environment, and random seeds with any reported experiment. The CLI uses seed 42 by default; pass `--seed` to select another value. A checkpoint stores the model configuration and weights; training history is written as a JSON sidecar. Results should be reported only after evaluation on a documented held-out set. A seed improves repeatability but does not guarantee bitwise-identical results across hardware and library versions.
