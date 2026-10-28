import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
			nn.Linear(dim, hidden),
			nn.GELU(),
			nn.Linear(hidden, hidden),
			nn.GELU(),
			nn.Linear(hidden, 1),
		)
        
    def forward(self, x):
        return self.net(x)
    
class TransformerBaseline(nn.Module):
    def __init__(self, dim, hidden, n_heads=4, n_tokens=8, n_layers=2):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = dim // n_tokens

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=n_heads,
                dim_feedforward=hidden,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        B = x.size(0)
        x_tokens = x.view(B, self.n_tokens, self.token_dim)
        for layer in self.layers:
            x_tokens = layer(x_tokens)
        x_flat = x_tokens.reshape(B, -1)
        x_flat = self.norm(x_flat)
        return self.out(x_flat)