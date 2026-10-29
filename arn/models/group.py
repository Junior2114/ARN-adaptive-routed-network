import torch
import torch.nn as nn


class Group(nn.Module):
    def __init__(self, dim, hidden, n_heads=4, n_tokens=8):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = dim // n_tokens
        self.attn = nn.MultiheadAttention(self.token_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.size(0)
        x_tokens = x.view(B, self.n_tokens, self.token_dim)
        attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens, need_weights=False)
        x_tokens = self.norm1(x_tokens + attn_out)
        x_flat = x_tokens.reshape(B, -1)
        x = self.norm2(x + self.ff(x_flat))
        return x