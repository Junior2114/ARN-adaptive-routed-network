import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, dim, n_groups, proj_dim=32):
        super().__init__()
        self.orig_proj = nn.Linear(dim, proj_dim)
        self.head = nn.Linear(dim + 1 + proj_dim, n_groups)

    def forward(self, best_state, budget, original_state):
        orig_feat = self.orig_proj(original_state)
        x = torch.cat([best_state, budget.unsqueeze(-1), orig_feat], dim=-1)
        logits = self.head(x)
        return F.softmax(logits, dim=-1)