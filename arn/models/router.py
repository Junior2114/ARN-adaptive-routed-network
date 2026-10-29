import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, dim, n_groups):
        super().__init__()
        self.head = nn.Linear(dim + 1, n_groups)

    def forward(self, state_with_budget):
        logits = self.head(state_with_budget)
        return F.softmax(logits, dim=-1)