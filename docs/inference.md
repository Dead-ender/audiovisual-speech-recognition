# Inference

Inference requires a checkpoint produced by `avsr train` and a CSV manifest with audio paths, video paths, and reference transcripts. See [the data format guide](data-format.md) for path and vocabulary rules.

## Command line

```bash
avsr predict --manifest data/test.csv --checkpoint checkpoints/avsr.pt --output predictions.json
avsr evaluate --manifest data/test.csv --checkpoint checkpoints/avsr.pt
```

`predict` writes each reference and greedy CTC prediction as JSON. `evaluate` reports loss, character error rate (CER), and word error rate (WER). Evaluate against a held-out split, ideally speaker-disjoint, and document the split and preprocessing before reporting metrics.

## Python API

```python
import torch
from torch.utils.data import DataLoader

from avsr.checkpoint import load_checkpoint
from avsr.data import ManifestDataset, collate_audio_visual
from avsr.decode import greedy_ctc_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, config, metadata = load_checkpoint("checkpoints/avsr.pt", device=device)
dataset = ManifestDataset("data/test.csv", config)
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_audio_visual)

with torch.inference_mode():
    for batch in loader:
        logits, lengths = model(
            batch["audio"].to(device),
            batch["video"].to(device),
            batch["audio_lengths"].to(device),
            batch["video_lengths"].to(device),
        )
        print(greedy_ctc_decode(logits, lengths))
```

This is an inference recipe, not a benchmark. The repository currently includes no trained checkpoint or measured accuracy. See [the model card](../MODEL_CARD.md) for status and limitations.
