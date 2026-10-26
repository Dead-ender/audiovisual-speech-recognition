# Contributing

Contributions that improve correctness, reproducibility, documentation, or data handling are welcome.

## Development setup

1. Install a matching PyTorch/TorchAudio pair for your CPU or CUDA environment using the [official selector](https://pytorch.org/get-started/locally/).
2. Install the package and development tools with `python -m pip install -e ".[dev]"`.
3. Run `python -m pytest` and `ruff check .` before opening a pull request.

## Pull requests

- Include a focused test for changed behavior.
- Keep datasets, participant recordings, generated outputs, and model checkpoints out of commits.
- Describe dataset licenses, split strategy, preprocessing assumptions, and exact evaluation setup when adding experiments.
- Report measurements only when they come from a reproducible, held-out evaluation; include the dataset and split rather than presenting a number alone.
- Keep notebook outputs cleared and distinguish synthetic demos from real-data experiments.
