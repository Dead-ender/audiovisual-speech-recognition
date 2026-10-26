"""CTC decoding helpers."""

import torch

from avsr.text import TextCodec


def greedy_ctc_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor | None = None,
    codec: TextCodec | None = None,
) -> list[str]:
    """Greedy CTC decode for time-major logits shaped ``(T, B, V)``."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape (time, batch, vocabulary)")
    codec = codec or TextCodec()
    predictions = logits.argmax(dim=-1).detach().cpu()
    if lengths is None:
        lengths = torch.full((predictions.shape[1],), predictions.shape[0])
    lengths = lengths.detach().cpu().tolist()
    decoded = []
    for batch_index, length in enumerate(lengths):
        output: list[int] = []
        previous = None
        for label in predictions[: int(length), batch_index].tolist():
            if label != previous and label != codec.blank_id:
                output.append(label)
            previous = label
        decoded.append(codec.decode(output))
    return decoded
