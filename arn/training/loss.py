import torch
import torch.nn.functional as F


def arn_loss(model, pred, y, halt_stack, traces, state_stack, balance_loss,
             aux_weight=0.7, halt_weight=0.01, balance_weight=0.01,
             cost_weight=0.0005, prior_lambda=0.2):
    halt_detached = halt_stack.detach()

    per_step_loss = 0.0
    for k in range(state_stack.size(0)):
        step_pred = model.out(state_stack[k])
        # smooth_l1_loss устойчив к выбросам
        step_loss = F.smooth_l1_loss(step_pred, y, reduction="none").squeeze(-1)
        per_step_loss = per_step_loss + halt_detached[k] * step_loss
    main_loss = per_step_loss.mean()

    steps = halt_stack.size(0)
    B = halt_stack.size(1)
    k_idx = torch.arange(steps, dtype=torch.float32)
    prior = prior_lambda * (1 - prior_lambda) ** k_idx
    prior = prior.unsqueeze(1).expand(steps, B)
    eps = 1e-8
    halt_loss = (halt_stack * (torch.log(halt_stack + eps) - torch.log(prior + eps))).sum(dim=0).mean()

    aux_loss = 0.0
    for t in traces:
        with torch.no_grad():
            # Ограничиваем target, чтобы aux_loss не взрывался
            score_target = -F.mse_loss(model.out(t["state"]), y, reduction="none").squeeze(-1)
            score_target = torch.clamp(score_target, -1.0, 0.0)
        
        predicted = model.progress_eval(t["state"])
        # Ограничиваем предсказание и используем smooth_l1
        predicted = torch.clamp(predicted, -5.0, 5.0)
        aux_loss = aux_loss + F.smooth_l1_loss(predicted, score_target, reduction="mean")
    
    aux_loss = aux_loss / len(traces)

    steps_idx = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    expected_steps = (steps_idx * halt_detached).sum(dim=0).mean()

    total = (main_loss + aux_weight * aux_loss + halt_weight * halt_loss
             + balance_weight * balance_loss + cost_weight * expected_steps)
    return total, main_loss, aux_loss, halt_loss, balance_loss, expected_steps