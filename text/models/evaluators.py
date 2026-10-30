import torch
import torch.nn as nn


class ProgressEvaluator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, state):
        pooled = state.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class ReadinessEvaluator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim + 1, 1)

    def forward(self, state, budget):
        pooled = state.mean(dim=1)
        x = torch.cat([pooled, budget.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.head(x)).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, budget):
        pooled = state.mean(dim=1)
        x = torch.cat([pooled, budget.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)