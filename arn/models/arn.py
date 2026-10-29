import torch
import torch.nn as nn

from arn.config import ARNConfig
from arn.models.group import Group
from arn.models.router import Router
from arn.models.evaluators import (
    ProgressEvaluator, ReadinessEvaluator, ValueNetwork,
)


class ARN(nn.Module):
    def __init__(self, dim, hidden, n_groups, temperature=0.2):
        super().__init__()
        self.groups = nn.ModuleList([
            Group(dim, hidden, n_heads=4, n_tokens=8)
            for _ in range(n_groups)
        ])
        self.router = Router(dim, n_groups)
        self.progress_eval = ProgressEvaluator(dim)
        self.readiness_eval = ReadinessEvaluator(dim)
        self.value_net = ValueNetwork(dim, hidden=hidden)
        self.out = nn.Linear(dim, 1)
        self.temperature = temperature
        self.n_groups = n_groups

    def forward(self, state, max_steps=8, budget_init=1.0):
        best_state = state.clone()
        best_progress = torch.full((state.size(0),), -1e6)
        budget = torch.full((state.size(0),), budget_init)
        budget_per_step = budget_init / max_steps

        not_stopped = torch.ones(state.size(0))
        halt_probs, states, traces, log_probs, entropies, values = [], [], [], [], [], []

        for step in range(max_steps):
            router_input = torch.cat([best_state, budget.unsqueeze(-1)], dim=-1)
            weights = self.router(router_input)

            dist = torch.distributions.Categorical(weights)
            group_idx = dist.sample()
            log_prob_group = dist.log_prob(group_idx)
            entropy_group = dist.entropy()

            outs = torch.stack([g(best_state) for g in self.groups], dim=1)
            candidate = outs[torch.arange(state.size(0)), group_idx]

            cand_progress = self.progress_eval(candidate)
            alpha = torch.sigmoid(
                (cand_progress - best_progress) / self.temperature
            ).unsqueeze(-1)
            best_state = alpha * candidate + (1 - alpha) * best_state
            best_progress = torch.maximum(best_progress, cand_progress).detach()

            readiness_input = torch.cat([best_state, budget.unsqueeze(-1)], dim=-1)
            p_stop = self.readiness_eval(readiness_input)

            if step == max_steps - 1:
                halt_prob = not_stopped
            else:
                halt_prob = p_stop * not_stopped

            halt_probs.append(halt_prob)
            states.append(best_state)
            log_probs.append(log_prob_group)
            entropies.append(entropy_group)

            value_input = torch.cat([best_state, budget.unsqueeze(-1)], dim=-1)
            value = self.value_net(value_input)
            values.append(value)

            not_stopped = not_stopped * (1 - p_stop)

            traces.append({
                "step": step, "state": candidate.detach(),
                "best_state": best_state.detach(), "weights": weights.detach(),
                "group_idx": group_idx.detach(), "alpha": alpha.detach(),
                "progress": cand_progress.detach(), "p_stop": p_stop.detach(),
                "halt_prob": halt_prob.detach(), "not_stopped": not_stopped.detach(),
                "budget": budget.detach(),
            })
            budget = budget - budget_per_step

        halt_stack = torch.stack(halt_probs, dim=0)
        state_stack = torch.stack(states, dim=0)
        final_state = (halt_stack.unsqueeze(-1) * state_stack).sum(dim=0)
        pred = self.out(final_state)

        log_probs_stack = torch.stack(log_probs, dim=0)
        entropies_stack = torch.stack(entropies, dim=0)
        values_stack = torch.stack(values, dim=0)

        return pred, traces, halt_stack, state_stack, log_probs_stack, entropies_stack, values_stack