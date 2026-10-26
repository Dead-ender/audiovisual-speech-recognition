"""Model and preprocessing configuration."""

from dataclasses import asdict, dataclass

from avsr.text import DEFAULT_CHARSET


@dataclass(frozen=True)
class AVSRConfig:
    """Configuration shared by feature extraction and the multimodal model."""

    sample_rate: int = 16_000
    n_mels: int = 80
    win_length: int = 400
    hop_length: int = 160
    n_frames: int = 32
    frame_height: int = 64
    frame_width: int = 64
    model_dim: int = 256
    num_heads: int = 4
    num_transformer_layers: int = 4
    dropout: float = 0.1
    charset: str = DEFAULT_CHARSET

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.n_mels <= 0:
            raise ValueError("sample_rate and n_mels must be positive")
        if self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("win_length and hop_length must be positive")
        if min(self.n_frames, self.frame_height, self.frame_width) <= 0:
            raise ValueError("video dimensions must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model_dim <= 0 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if self.model_dim % 2:
            raise ValueError("model_dim must be even for sinusoidal positions")
        if self.num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if len(set(self.charset)) != len(self.charset):
            raise ValueError("charset characters must be unique")

    @property
    def blank_id(self) -> int:
        """CTC reserves class zero for the blank symbol."""
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.charset) + 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
