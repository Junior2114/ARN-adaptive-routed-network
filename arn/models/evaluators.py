import torch
import torch.nn as nn


class ProgressEvaluator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, state):
        return self.head(state).squeeze(-1)


class ReadinessEvaluator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim + 1, 1)

    def forward(self, state_with_budget):
        return torch.sigmoid(self.head(state_with_budget)).squeeze(-1)


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

    def forward(self, state_with_budget):
        return self.net(state_with_budget).squeeze(-1)