import torch
import torch.nn as nn


class Group(nn.Module):
    def __init__(self, dim, hidden, n_heads=4, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=hidden,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, dim]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)