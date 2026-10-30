import time
import os
import torch
import torch.nn.functional as F

from arn.models.arn import ARN
from arn.training.loss import arn_loss
from arn.training.ppo import (
    compute_gae, normalize_advantages, ppo_loss, compute_approx_kl,
)


def train_model(model, train_data, test_data, epochs=200, lr=1e-3,
                lr_router=1e-4, lr_value=1e-3, aux_weight=0.7,
                halt_weight=0.01, balance_weight=0.01, cost_weight=0.0005,
                prior_lambda=0.2, ppo_epochs=3, ppo_warmup=20,
                gamma=0.95, lam=0.95, clip_eps=0.2, entropy_coef=0.03,
                target_kl=0.05, max_grad_norm=0.5, weight_decay=1e-4,
                log_every=20, save_path="checkpoints/model.pt",
                load_path=None, patience=20, min_delta=1e-5):
    x_train, y_train = train_data
    x_test, y_test = test_data

    start_epoch = 0
    if load_path and os.path.exists(load_path):
        print(f"Loading checkpoint from {load_path}...")
        checkpoint = torch.load(load_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    history = {"train": [], "test": [], "aux": [], "halt": [],
               "balance": [], "steps": [], "policy": [], "value": []}
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    start_time = time.time()

    best_test = float("inf")
    best_state_dict = None
    early_stop_counter = 0
    last_epoch = start_epoch

    def check_early_stop(test_loss, epoch):
        nonlocal best_test, best_state_dict, early_stop_counter
        if test_loss < best_test - min_delta:
            best_test = test_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best test: {best_test:.6f}")
            if best_state_dict:
                model.load_state_dict(best_state_dict)
                print("Loaded best model weights.")
            return True
        return False

    # Если epochs=0, просто сохраняем и выходим
    if epochs == 0:
        torch.save({"model_state_dict": model.state_dict(), "epoch": start_epoch}, save_path)
        print(f"Saved checkpoint to {save_path} (no training)")
        return history

    if not isinstance(model, ARN):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(start_epoch, epochs):
            last_epoch = epoch
            epoch_start = time.time()
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
            
            if check_early_stop(test_loss.item(), epoch):
                break

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            eta = (total_time / (epoch - start_epoch + 1)) * (epochs - epoch - 1)

            if epoch % log_every == 0:
                print(f"epoch={epoch:4d} train={loss.item():.4f} test={test_loss.item():.4f} "
                      f"| {epoch_time:.2f}s (ETA: {eta/60:.1f}m)")

        torch.save({"model_state_dict": model.state_dict(), "epoch": last_epoch}, save_path)
        print(f"\n[Baseline] Total time: {time.time() - start_time:.2f}s")
        print(f"Saved checkpoint to {save_path}")
        return history

    # Soft routing
    if model.routing_mode == "soft":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(start_epoch, epochs):
            last_epoch = epoch
            epoch_start = time.time()
            model.train()
            (pred, traces, halt_stack, state_stack,
             _, _, _, balance_loss) = model(x_train)
            (loss, main_loss, aux_loss, halt_loss,
             balance_loss, expected_steps) = arn_loss(
                model, pred, y_train, halt_stack, traces, state_stack, balance_loss,
                aux_weight=aux_weight, halt_weight=halt_weight,
                balance_weight=balance_weight, cost_weight=cost_weight,
                prior_lambda=prior_lambda,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                pred_test, _, halt_stack_test, _, _, _, _, _ = model(x_test)
                test_loss = F.mse_loss(pred_test, y_test)
                steps_val = (torch.arange(halt_stack_test.size(0)).float().unsqueeze(1) * halt_stack_test).sum(dim=0).mean().item()

            history["train"].append(loss.item())
            history["test"].append(test_loss.item())
            history["aux"].append(aux_loss.item())
            history["halt"].append(halt_loss.item())
            history["balance"].append(balance_loss.item())
            history["steps"].append(steps_val)
            history["policy"].append(0.0)
            history["value"].append(0.0)

            if check_early_stop(test_loss.item(), epoch):
                break

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            eta = (total_time / (epoch - start_epoch + 1)) * (epochs - epoch - 1)

            if epoch % log_every == 0:
                print(f"epoch={epoch:4d} sup={loss.item():.4f} aux={aux_loss.item():.4f} "
                      f"test={test_loss.item():.4f} | {epoch_time:.2f}s (ETA: {eta/60:.1f}m)")

        torch.save({"model_state_dict": model.state_dict(), "epoch": last_epoch}, save_path)
        print(f"\n[Soft] Total time: {time.time() - start_time:.2f}s")
        print(f"Saved checkpoint to {save_path}")
        return history

    # Discrete routing (PPO)
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

    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        epoch_start = time.time()
        model.train()

        (pred, traces, halt_stack, state_stack,
         log_probs, entropies, values, balance_loss) = model(x_train)

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

        policy_loss_val = 0.0
        value_loss_val = 0.0
        
        if epoch >= ppo_warmup:
            with torch.no_grad():
                rewards = []
                for k in range(state_stack.size(0)):
                    step_pred = model.out(state_stack[k])
                    step_loss = F.smooth_l1_loss(step_pred, y_train, reduction="none").squeeze(-1)
                    rewards.append(-step_loss)
                rewards = torch.stack(rewards, dim=0)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                dones = torch.zeros_like(rewards)

                advantages, returns = compute_gae(rewards, values.detach(), dones, gamma, lam)
                advantages = normalize_advantages(advantages)
                returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

            value_loss = 0.0
            for k in range(state_stack.size(0)):
                state_k = state_stack[k].detach()
                budget_k = traces[k]["budget"].detach()
                value_input = torch.cat([state_k, budget_k.unsqueeze(-1)], dim=-1)
                v_pred = model.value_net(value_input)
                value_loss = value_loss + F.smooth_l1_loss(v_pred, returns_norm[k])
            value_loss = value_loss / state_stack.size(0)
            
            opt_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.value_net.parameters(), max_norm=max_grad_norm)
            opt_value.step()
            value_loss_val = value_loss.item()

            for _ in range(ppo_epochs):
                (pred, traces, halt_stack, state_stack,
                 log_probs_new, entropies_new, values_new, _) = model(x_train)
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
            pred_test, _, halt_stack_test, _, _, _, _, _ = model(x_test)
            test_loss = F.mse_loss(pred_test, y_test)
            steps_val = (torch.arange(halt_stack_test.size(0)).float().unsqueeze(1) * halt_stack_test).sum(dim=0).mean().item()

        history["train"].append(supervised_loss.item())
        history["test"].append(test_loss.item())
        history["aux"].append(aux_loss.item())
        history["halt"].append(halt_loss.item())
        history["balance"].append(balance_loss.item())
        history["steps"].append(steps_val)
        history["policy"].append(policy_loss_val)
        history["value"].append(value_loss_val)

        if check_early_stop(test_loss.item(), epoch):
            break

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        eta = (total_time / (epoch - start_epoch + 1)) * (epochs - epoch - 1)

        if epoch % log_every == 0:
            print(f"epoch={epoch:4d} sup={supervised_loss.item():.4f} "
                  f"aux={aux_loss.item():.4f} val={value_loss_val:.4f} "
                  f"pol={policy_loss_val:.4f} steps={steps_val:.2f} "
                  f"test={test_loss.item():.4f} | {epoch_time:.2f}s (ETA: {eta/60:.1f}m)")

    torch.save({"model_state_dict": model.state_dict(), "epoch": last_epoch}, save_path)
    print(f"\n[PPO] Total time: {time.time() - start_time:.2f}s")
    print(f"Saved checkpoint to {save_path}")
    return history