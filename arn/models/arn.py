import torch
import torch.nn as nn
import torch.nn.functional as F

from arn.models.group import Group
from arn.models.router import Router
from arn.models.evaluators import (
    ProgressEvaluator, ReadinessEvaluator, ValueNetwork,
)


class ARN(nn.Module):
    def __init__(self, dim, hidden, n_groups, temperature=0.2, routing_mode="soft"):
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
        self.routing_mode = routing_mode

    def forward(self, state, max_steps=8, budget_init=1.0):
        original_state = state
        best_state = state.clone()
        best_progress = torch.full((state.size(0),), -1e6)
        budget = torch.full((state.size(0),), budget_init)
        budget_per_step = budget_init / max_steps

        not_stopped = torch.ones(state.size(0))
        halt_probs, states, traces, all_weights = [], [], [], []
        log_probs_list, entropies_list, values_list = [], [], []

        for step in range(max_steps):
            weights = self.router(best_state, budget, original_state)
            all_weights.append(weights)

            outs = torch.stack([g(best_state) for g in self.groups], dim=1)

            if self.routing_mode == "discrete":
                dist = torch.distributions.Categorical(weights)
                group_idx = dist.sample()
                log_prob_group = dist.log_prob(group_idx)
                entropy_group = dist.entropy()
                candidate = outs[torch.arange(state.size(0)), group_idx]
                log_probs_list.append(log_prob_group)
                entropies_list.append(entropy_group)
            else:
                candidate = (weights.unsqueeze(-1) * outs).sum(dim=1)
                group_idx = None

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
            not_stopped = not_stopped * (1 - p_stop)

            value_input = torch.cat([best_state, budget.unsqueeze(-1)], dim=-1)
            values_list.append(self.value_net(value_input))

            traces.append({
                "step": step, "state": candidate.detach(),
                "best_state": best_state.detach(), "weights": weights.detach(),
                "alpha": alpha.detach(), "progress": cand_progress.detach(),
                "p_stop": p_stop.detach(), "halt_prob": halt_prob.detach(),
                "not_stopped": not_stopped.detach(), "budget": budget.detach(),
            })
            budget = budget - budget_per_step

        weights_stack = torch.stack(all_weights, dim=0)
        avg_weights = weights_stack.mean(dim=(0, 1))
        target = torch.ones_like(avg_weights) / avg_weights.size(0)
        balance_loss = F.mse_loss(avg_weights, target)

        halt_stack = torch.stack(halt_probs, dim=0)
        state_stack = torch.stack(states, dim=0)
        final_state = (halt_stack.unsqueeze(-1) * state_stack).sum(dim=0)
        pred = self.out(final_state)

        log_probs_stack = torch.stack(log_probs_list, dim=0) if log_probs_list else None
        entropies_stack = torch.stack(entropies_list, dim=0) if entropies_list else None
        values_stack = torch.stack(values_list, dim=0)

        return (pred, traces, halt_stack, state_stack,
                log_probs_stack, entropies_stack, values_stack, balance_loss)