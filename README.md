# Audiovisual Speech Recognition Prototype

A small research prototype exploring multimodal fusion for character-level speech recognition. It combines a log-Mel spectrogram audio encoder, a 3D CNN over grayscale video frames, a Transformer fusion encoder, and a CTC output layer. The implementation and runnable synthetic demo are in the notebook.

## Status and limitations

This is a pipeline demonstration, not a trained speech-recognition system. Its bundled dataset is generated in memory from synthetic sine waves/noise, moving squares, and random character sequences. The inputs and labels are unrelated and contain no real speech or lip motion. There is no real-world dataset, validation protocol, benchmark, or reported accuracy. Training output from the demo must not be interpreted as recognition performance.

The notebook currently expects fixed-size synthetic clips and is intended for experimentation with model wiring. It does not yet include real-media preprocessing, alignment, checkpointing, evaluation metrics, or inference on user-provided recordings.

## Requirements

- Python 3.10 or newer
- PyTorch and TorchAudio built for the same release and compatible CPU/CUDA runtime
- NumPy below 2.0 (needed for compatibility with older PyTorch builds)

For GPU installations, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to install matching `torch` and `torchaudio` packages for your platform. Then install the remaining requirements, or install all packages from `requirements.txt` in an environment where pip can select compatible builds.

```bash
python -m venv .venv
# Activate the environment, then install a matching torch/torchaudio pair.
python -m pip install -r requirements.txt
```

## Run the notebook

Open `Audiovisual Speech Recognition through Multimodal Fusion of Lip Movements and Audio Spectrograms.ipynb` in Jupyter or Google Colab. The final cell leaves `RUN_DEMO = False` by default. Change it to `True` and run the cell to execute the small synthetic training demo. A GPU is optional; CPU execution may be slow.

## Tests

The lightweight tests validate notebook structure, Python syntax, and that generated cell outputs are not committed. They do not run model training or assess recognition quality.

```bash
python -m pip install pytest
python -m pytest
```

## Repository contents

- `Audiovisual Speech Recognition through Multimodal Fusion of Lip Movements and Audio Spectrograms.ipynb` — model prototype and opt-in synthetic demo
- `requirements.txt` — Python dependencies
- `tests/` — notebook integrity checks
