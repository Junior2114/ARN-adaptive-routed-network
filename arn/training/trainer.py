import torch
import torch.nn.functional as F

from arn.models.arn import ARN
from arn.training.loss import arn_loss
from arn.training.ppo import (
    compute_gae, normalize_advantages, ppo_loss, compute_approx_kl,
)


def train_model(model, train_data, test_data, epochs=200, lr=1e-3,
                lr_router=3e-4, lr_value=1e-3, aux_weight=0.7,
                halt_weight=0.01, balance_weight=0.01, cost_weight=0.0005,
                prior_lambda=0.2, ppo_epochs=3, gamma=0.95, lam=0.95,
                clip_eps=0.2, entropy_coef=0.03, target_kl=0.05,
                max_grad_norm=1.0, weight_decay=1e-4, log_every=20):
    x_train, y_train = train_data
    x_test, y_test = test_data

    history = {"train": [], "test": [], "aux": [], "halt": [],
               "balance": [], "steps": [], "policy": [], "value": []}

    # Baseline
    if not isinstance(model, ARN):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            pred = model(x_train)
            loss = F.mse_loss(pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pred_test = model(x_test)
                test_loss = F.mse_loss(pred_test, y_test)
            history["train"].append(loss.item())
            history["test"].append(test_loss.item())
            for key in ["aux", "halt", "balance", "steps", "policy", "value"]:
                history[key].append(0.0)
            if epoch % log_every == 0:
                print(f"epoch={epoch:4d} train={loss.item():.4f} test={test_loss.item():.4f}")
        return history

    # ARN: PPO
    opt_groups = torch.optim.Adam([
        {"params": list(model.groups.parameters()), "weight_decay": weight_decay},
        {"params": list(model.out.parameters())},
        {"params": list(model.progress_eval.parameters())},
    ], lr=lr)

    opt_router = torch.optim.Adam(
        list(model.router.parameters()) + list(model.readiness_eval.parameters()),
        lr=lr_router, betas=(0.9, 0.95),
    )
    opt_value = torch.optim.Adam(model.value_net.parameters(), lr=lr_value)

    for epoch in range(epochs):
        model.train()

        with torch.no_grad():
            (pred, traces, halt_stack, state_stack,
             log_probs, entropies, values) = model(x_train)

        rewards = []
        for k in range(state_stack.size(0)):
            step_pred = model.out(state_stack[k])
            step_loss = F.mse_loss(step_pred, y_train, reduction="none").squeeze(-1)
            rewards.append(-step_loss)
        rewards = torch.stack(rewards, dim=0)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones = torch.zeros_like(rewards)

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        advantages = normalize_advantages(advantages)
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns_norm = returns_norm.detach()

        (pred, traces, halt_stack, state_stack,
         log_probs_new, entropies, values_new) = model(x_train)

        weights_stack = torch.stack([t["weights"] for t in traces])
        avg_weights = weights_stack.mean(dim=(0, 1))
        target = torch.ones_like(avg_weights) / avg_weights.size(0)
        balance_loss = F.mse_loss(avg_weights, target)

        (supervised_loss, main_loss, aux_loss, halt_loss,
         balance_loss, expected_steps) = arn_loss(
            model, pred, y_train, halt_stack, traces, state_stack, balance_loss,
            aux_weight=aux_weight, halt_weight=halt_weight,
            balance_weight=balance_weight, cost_weight=cost_weight,
            prior_lambda=prior_lambda,
        )

        opt_groups.zero_grad()
        supervised_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.groups.parameters(), max_norm=max_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.out.parameters(), max_norm=max_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.progress_eval.parameters(), max_norm=max_grad_norm)
        opt_groups.step()

        value_loss = 0.0
        for k in range(state_stack.size(0)):
            state_k = state_stack[k].detach()
            budget_k = traces[k]["budget"].detach()
            value_input = torch.cat([state_k, budget_k.unsqueeze(-1)], dim=-1)
            v_pred = model.value_net(value_input)
            value_loss = value_loss + F.mse_loss(v_pred, returns_norm[k])
        value_loss = value_loss / state_stack.size(0)

        opt_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.value_net.parameters(), max_norm=max_grad_norm)
        opt_value.step()

        policy_loss_val = 0.0
        for _ in range(ppo_epochs):
            (pred, traces, halt_stack, state_stack,
             log_probs_new, entropies_new, values_new) = model(x_train)
            policy_loss = ppo_loss(log_probs_new, log_probs.detach(),
                                   advantages, entropies_new,
                                   clip_eps, entropy_coef)
            opt_router.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), max_norm=max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.readiness_eval.parameters(), max_norm=max_grad_norm)
            opt_router.step()
            policy_loss_val = policy_loss.item()

            approx_kl = compute_approx_kl(log_probs_new, log_probs.detach()).item()
            if approx_kl > target_kl:
                break

        model.eval()
        with torch.no_grad():
            pred_test, _, halt_stack_test, _, _, _, _ = model(x_test)
            test_loss = F.mse_loss(pred_test, y_test)
            steps_val = (torch.arange(halt_stack_test.size(0)).float().unsqueeze(1) * halt_stack_test).sum(dim=0).mean().item()

        history["train"].append(supervised_loss.item())
        history["test"].append(test_loss.item())
        history["aux"].append(aux_loss.item())
        history["halt"].append(halt_loss.item())
        history["balance"].append(balance_loss.item())
        history["steps"].append(steps_val)
        history["policy"].append(policy_loss_val)
        history["value"].append(value_loss.item())

        if epoch % log_every == 0:
            print(f"epoch={epoch:4d} sup={supervised_loss.item():.4f} "
                  f"val={value_loss.item():.4f} pol={policy_loss_val:.4f} "
                  f"steps={steps_val:.2f} test={test_loss.item():.4f}")

    return history