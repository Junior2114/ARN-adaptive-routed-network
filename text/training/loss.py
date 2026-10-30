import torch
import torch.nn.functional as F


def text_arn_loss(model, logits, targets, halt_stack, traces, state_stack,
                  aux_weight=0.5, halt_weight=0.01, prior_lambda=0.2):
    B, T, V = logits.shape

    # Main loss: cross-entropy по токенам
    main_loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

    # Halt loss (KL к prior)
    steps = halt_stack.size(0)
    k_idx = torch.arange(steps, device=halt_stack.device, dtype=torch.float32)
    prior = prior_lambda * (1 - prior_lambda) ** k_idx
    prior = prior.unsqueeze(1).expand(steps, B)
    eps = 1e-8
    halt_loss = (halt_stack * (torch.log(halt_stack + eps) - torch.log(prior + eps))).sum(dim=0).mean()

    # Aux loss для progress evaluator
    aux_loss = 0.0
    for t in traces:
        with torch.no_grad():
            pooled = t["state"].mean(dim=1)
            # target: -CE для текущего состояния
            step_logits = model.out(t["state"])
            score_target = -F.cross_entropy(
                step_logits.reshape(-1, V), targets.reshape(-1), reduction="none"
            ).reshape(B, T).mean(dim=1)
        predicted = model.progress_eval(t["state"])
        aux_loss = aux_loss + F.mse_loss(predicted, score_target)
    aux_loss = aux_loss / len(traces)

    total = main_loss + aux_weight * aux_loss + halt_weight * halt_loss
    return total, main_loss, aux_loss, halt_loss