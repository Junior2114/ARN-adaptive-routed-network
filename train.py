import torch
import torch.nn.functional as F

from arn import ARN


def arn_loss(model, pred, y, halt_stack, traces, state_stack, balance_loss,
             aux_weight=0.5, halt_weight=0.01, balance_weight=0.01,
             cost_weight=0.0005, prior_lambda=0.2):
    halt_stack_detached = halt_stack.detach()
    
    per_step_loss = 0.0
    for k in range(state_stack.size(0)):
        step_pred = model.out(state_stack[k])
        step_loss = F.mse_loss(step_pred, y, reduction="none").squeeze(-1)
        per_step_loss = per_step_loss + halt_stack_detached[k] * step_loss
    main_loss = per_step_loss.mean()

    steps = halt_stack.size(0)
    B = halt_stack.size(1)
    k_idx = torch.arange(steps, dtype=torch.float32)
    prior = prior_lambda * (1 - prior_lambda) ** k_idx
    prior = prior.unsqueeze(1).expand(steps, B)

    eps = 1e-8
    halt_loss = (
        halt_stack * (torch.log(halt_stack + eps) - torch.log(prior + eps))
    ).sum(dim=0).mean()

    aux_loss = 0.0
    for t in traces:
        state = t["state"]
        with torch.no_grad():
            score_target = -F.mse_loss(
                model.out(state), y, reduction="none"
            ).squeeze(-1)
        predicted = model.progress_eval(state)
        aux_loss = aux_loss + F.mse_loss(predicted, score_target)
    aux_loss = aux_loss / len(traces)

    steps_idx = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    expected_steps = (steps_idx * halt_stack_detached).sum(dim=0).mean()

    total = (main_loss
             + aux_weight * aux_loss
             + halt_weight * halt_loss
             + balance_weight * balance_loss
             + cost_weight * expected_steps)

    return total, main_loss, aux_loss, halt_loss, balance_loss, expected_steps


def train_model(model, train_data, test_data, epochs=200, lr=1e-3,
                aux_weight=0.5, halt_weight=0.01, balance_weight=0.01,
                cost_weight=0.0005, prior_lambda=0.2, log_every=20):
    x_train, y_train = train_data
    x_test, y_test = test_data

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "test": [], "aux": [], "halt": [],
               "balance": [], "steps": []}

    for epoch in range(epochs):
        model.train()
        if isinstance(model, ARN):
            pred, traces, halt_stack, state_stack, balance_loss = model(x_train)
            (loss, main_loss, aux_loss, halt_loss,
             balance_loss, expected_steps) = arn_loss(
                model, pred, y_train, halt_stack, traces, state_stack, balance_loss,
                aux_weight=aux_weight, halt_weight=halt_weight,
                balance_weight=balance_weight, cost_weight=cost_weight,
                prior_lambda=prior_lambda,
            )
        else:
            pred = model(x_train)
            loss = F.mse_loss(pred, y_train)
            aux_loss = torch.tensor(0.0)
            halt_loss = torch.tensor(0.0)
            balance_loss = torch.tensor(0.0)
            expected_steps = torch.tensor(0.0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            if isinstance(model, ARN):
                pred_test, _, _, _, _ = model(x_test)
            else:
                pred_test = model(x_test)
            test_loss = F.mse_loss(pred_test, y_test)

        history["train"].append(loss.item())
        history["test"].append(test_loss.item())
        history["aux"].append(aux_loss.item())
        history["halt"].append(halt_loss.item())
        history["balance"].append(balance_loss.item())
        history["steps"].append(expected_steps.item())

        if epoch % log_every == 0:
            print(f"epoch={epoch:4d} train={loss.item():.4f} "
                  f"aux={aux_loss.item():.4f} halt={halt_loss.item():.4f} "
                  f"bal={balance_loss.item():.4f} steps={expected_steps.item():.2f} "
                  f"test={test_loss.item():.4f}")

    return history


def diagnose(model, x, y=None):
    model.eval()
    with torch.no_grad():
        if isinstance(model, ARN):
            _, trace, halt_stack, _, _ = model(x)
        else:
            return

    steps = torch.arange(halt_stack.size(0), dtype=torch.float32).unsqueeze(1)
    expected_steps = (steps * halt_stack).sum(dim=0)

    print("Распределение ожидаемых шагов:")
    for k in range(halt_stack.size(0)):
        count = ((expected_steps >= k) & (expected_steps < k + 1)).sum().item()
        if count > 0:
            print(f"  ~{k} шагов: {count} примеров")

    print(f"\nСреднее число шагов: {expected_steps.mean():.2f}")
    print(f"Максимум: {expected_steps.max():.2f}")
    print(f"Минимум: {expected_steps.min():.2f}")

    print("\nСредняя halt_prob по шагам:")
    for s in range(halt_stack.size(0)):
        hp = halt_stack[s].mean().item()
        print(f"  step {s}: {hp:.3f}")

    weights = torch.stack([t["weights"] for t in trace])
    alpha = torch.stack([t["alpha"] for t in trace])
    progress = torch.stack([t["progress"] for t in trace])
    p_stop = torch.stack([t["p_stop"] for t in trace])

    print("\nСредние веса групп по шагам:")
    for s in range(weights.shape[0]):
        w = weights[s].mean(dim=0)
        print(f"  step {s}: {[round(v, 3) for v in w.tolist()]}")

    print("\nСредний alpha по шагам:")
    for s in range(alpha.shape[0]):
        a = alpha[s].mean().item()
        print(f"  step {s}: {a:.3f}")

    print("\nСредний progress по шагам:")
    for s in range(progress.shape[0]):
        p = progress[s].mean().item()
        print(f"  step {s}: {p:.4f}")

    print("\nСредний readiness (p_stop) по шагам:")
    for s in range(p_stop.shape[0]):
        r = p_stop[s].mean().item()
        print(f"  step {s}: {r:.3f}")

    avg_weights = weights.mean(dim=(0, 1))
    print(f"\nСредние веса групп (по всем шагам): "
          f"{[round(v, 3) for v in avg_weights.tolist()]}")
    print(f"Отклонение от равномерности: {avg_weights.std().item():.4f}")

    if y is not None:
        print("\nПрямая оценка состояний (без Evaluator):")
        for s, t in enumerate(trace):
            state = t["state"]
            true_score = -F.mse_loss(
                model.out(state), y, reduction="none"
            ).squeeze(-1)
            print(f"  step {s}: true={true_score.mean():.4f}")