import torch
import torch.nn.functional as F

from arn.models.arn import ARN


def diagnose(model, x, y=None):
    model.eval()
    with torch.no_grad():
        if isinstance(model, ARN):
            _, trace, halt_stack, _, _, _, _ = model(x)
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
            true_score = -F.mse_loss(model.out(state), y, reduction="none").squeeze(-1)
            print(f"  step {s}: true={true_score.mean():.4f}")