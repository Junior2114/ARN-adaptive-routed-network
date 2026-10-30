import torch
import torch.nn as nn
import torch.nn.functional as F

from text.models.group import Group
from text.models.router import Router
from text.models.evaluators import (
    ProgressEvaluator, ReadinessEvaluator, ValueNetwork,
)


class TextARN(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim, hidden, n_groups,
                 max_steps=4, temperature=0.2, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        self.groups = nn.ModuleList([
            Group(dim, hidden, n_heads=n_heads, n_layers=n_layers)
            for _ in range(n_groups)
        ])
        self.router = Router(dim, n_groups)
        self.progress_eval = ProgressEvaluator(dim)
        self.readiness_eval = ReadinessEvaluator(dim)
        self.value_net = ValueNetwork(dim, hidden=hidden)

        self.out = nn.Linear(dim, vocab_size)
        self.temperature = temperature
        self.max_steps = max_steps

    def forward(self, input_ids, budget_init=1.0):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        # Начальное состояние: эмбеддинги
        state = self.embedding(input_ids) + self.pos_embedding(pos)  # [B, T, dim]
        best_state = state.clone()
        best_progress = torch.full((B,), -1e6, device=input_ids.device)

        budget = torch.full((B,), budget_init, device=input_ids.device)
        budget_per_step = budget_init / self.max_steps

        not_stopped = torch.ones(B, device=input_ids.device)
        halt_probs, states, traces = [], [], []

        for step in range(self.max_steps):
            weights = self.router(best_state, budget)  # [B, G]

            # Soft mixture групп
            outs = torch.stack([g(best_state) for g in self.groups], dim=1)  # [B, G, T, D]
            candidate = (weights.unsqueeze(-1).unsqueeze(-1) * outs).sum(dim=1)  # [B, T, D]

            cand_progress = self.progress_eval(candidate)  # [B]
            alpha = torch.sigmoid(
                (cand_progress - best_progress) / self.temperature
            ).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            best_state = alpha * candidate + (1 - alpha) * best_state
            best_progress = torch.maximum(best_progress, cand_progress).detach()

            p_stop = self.readiness_eval(best_state, budget)  # [B]

            if step == self.max_steps - 1:
                halt_prob = not_stopped
            else:
                halt_prob = p_stop * not_stopped

            halt_probs.append(halt_prob)
            states.append(best_state)
            not_stopped = not_stopped * (1 - p_stop)

            traces.append({
                "step": step,
                "state": candidate.detach(),
                "best_state": best_state.detach(),
                "weights": weights.detach(),
                "alpha": alpha.detach(),
                "progress": cand_progress.detach(),
                "p_stop": p_stop.detach(),
                "halt_prob": halt_prob.detach(),
                "not_stopped": not_stopped.detach(),
                "budget": budget.detach(),
            })

            budget = budget - budget_per_step

        halt_stack = torch.stack(halt_probs, dim=0)  # [steps, B]
        state_stack = torch.stack(states, dim=0)      # [steps, B, T, D]

        final_state = (halt_stack.unsqueeze(-1).unsqueeze(-1) * state_stack).sum(dim=0)
        logits = self.out(final_state)  # [B, T, vocab]

        return logits, traces, halt_stack, state_stack