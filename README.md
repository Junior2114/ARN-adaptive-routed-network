# ARN — Adaptive Routed Network

Adaptive computation with attention groups, soft rollback, PonderNet halt loss, budget, and dual routing modes.

## Results

| Task | Baseline (big) | Transformer | ARN (best mode) |
|---|---|---|---|
| mixed | 0.001353 | 0.001375 | **0.000386** (Discrete/PPO) |
| conditional2 | 0.006897 | 0.008021 | **0.003121** (Soft) |

Training: 2000 samples, 400 test, 200 epochs, seed 0.

## Architecture Highlights
- **Dual routing**: Soft (mixture) for composition tasks, Discrete (PPO) for selection tasks.
- **Stabilization**: PPO Warmup, `smooth_l1_loss`, gradient clipping.
- **Adaptive depth**: Dynamic steps via Readiness Evaluator.
- **Routing specialization**: Groups dynamically shift weights across steps.