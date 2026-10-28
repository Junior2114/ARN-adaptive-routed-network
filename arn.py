import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 64
HIDDEN = 128
N_GROUPS = 4
MAX_STEPS = 8


class Group(nn.Module):
    def __init__(self, dim, hidden, n_heads=4, n_tokens=8):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = dim // n_tokens

        self.attn = nn.MultiheadAttention(
            self.token_dim, n_heads, batch_first=True
        )
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


class Router(nn.Module):
    def __init__(self, dim, n_groups):
        super().__init__()
        self.head = nn.Linear(dim + 1, n_groups)

    def forward(self, state_with_budget):
        logits = self.head(state_with_budget)
        return F.softmax(logits, dim=-1)


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


class ARN(nn.Module):
    def __init__(self, dim, hidden, n_groups, temperature=0.1):
        super().__init__()
        self.groups = nn.ModuleList([
            Group(dim, hidden, n_heads=4, n_tokens=8)
            for _ in range(n_groups)
        ])
        self.router = Router(dim, n_groups)
        self.progress_eval = ProgressEvaluator(dim)
        self.readiness_eval = ReadinessEvaluator(dim)
        self.out = nn.Linear(dim, 1)
        self.temperature = temperature

    def forward(self, state, max_steps=MAX_STEPS, budget_init=1.0):
        best_state = state.clone()
        best_progress = torch.full((state.size(0),), -1e6)

        budget = torch.full((state.size(0),), budget_init)
        budget_per_step = budget_init / max_steps

        not_stopped = torch.ones(state.size(0))
        halt_probs = []
        states = []
        traces = []
        all_weights = []

        for step in range(max_steps):
            router_input = torch.cat([best_state, budget.unsqueeze(-1)], dim=-1)
            weights = self.router(router_input)
            all_weights.append(weights)

            outs = torch.stack([g(best_state) for g in self.groups], dim=1)
            candidate = (weights.unsqueeze(-1) * outs).sum(dim=1)

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

        weights_stack = torch.stack(all_weights, dim=0)
        avg_weights = weights_stack.mean(dim=(0, 1))
        n_groups = avg_weights.size(0)
        target = torch.ones_like(avg_weights) / n_groups
        balance_loss = F.mse_loss(avg_weights, target)

        halt_stack = torch.stack(halt_probs, dim=0)
        state_stack = torch.stack(states, dim=0)

        final_state = (halt_stack.unsqueeze(-1) * state_stack).sum(dim=0)
        pred = self.out(final_state)

        return pred, traces, halt_stack, state_stack, balance_loss