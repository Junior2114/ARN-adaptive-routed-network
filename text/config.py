from dataclasses import dataclass


@dataclass
class TextARNConfig:
    vocab_size: int = 62
    max_seq_len: int = 64
    dim: int = 256
    hidden: int = 512
    n_groups: int = 8
    max_steps: int = 8
    temperature: float = 0.5
    n_heads: int = 4
    n_layers: int = 4