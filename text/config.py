from dataclasses import dataclass


@dataclass
class TextARNConfig:
    vocab_size: int = 65
    max_seq_len: int = 64
    dim: int = 128
    hidden: int = 256
    n_groups: int = 4
    max_steps: int = 8
    temperature: float = 0.5
    n_heads: int = 4
    n_layers: int = 2