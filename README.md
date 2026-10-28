# ARN — Adaptive Routed Network

Adaptive computation with attention groups, soft rollback, PonderNet halt loss, budget, and PPO-trained discrete router.

## Results

| Task | Baseline (big) | Transformer | ARN (PPO) |
|---|---|---|---|
| mixed | 0.001353 | 0.001375 | **0.000426** |
| conditional2 | 0.006897 | 0.008021 | **0.003918** |

Test set: 400 examples. Train set: 2000 examples. Seed: 0.

## Architecture

- **Groups**: attention over tokens (`MultiheadAttention` + FFN + LayerNorm)
- **Router**: discrete (`Categorical`) — chooses one group per step
- **Soft rollback**: `alpha = sigmoid((progress - best) / temperature)`, `temperature = 0.2`
- **Readiness**: `p_stop` for adaptive depth
- **Budget**: decreases per step, visible to Router and Readiness
- **PPO**: GAE (`gamma=0.95`, `lam=0.95`), advantage normalization, clipped objective (`clip_eps=0.2`), KL early stopping (`target_kl=0.05`)
- **Value network**: separate from progress evaluator, return normalization
- **Progress evaluator**: trained via aux loss, used for soft rollback

## Install

```bash
pip install -r requirements.txt