import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, dim, n_groups):
        super().__init__()
        self.head = nn.Linear(dim + 1, n_groups)

    def forward(self, state, budget):
        # state: [B, T, dim] → mean по T
        pooled = state.mean(dim=1)  # [B, dim]
        x = torch.cat([pooled, budget.unsqueeze(-1)], dim=-1)
        logits = self.head(x)
        return F.softmax(logits, dim=-1)