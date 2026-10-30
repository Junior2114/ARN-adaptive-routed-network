# ARN — Adaptive Routed Network

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ARN** is a neural architecture with adaptive computation depth and dynamic routing. It decides how many processing steps to take and which computational blocks to use, based on the input's complexity. It combines attention-based groups, soft rollback, PonderNet-style halt loss, and a budget mechanism with two routing modes: **Soft** (for composition tasks) and **Discrete/PPO** (for selection tasks).

## Results

| Task | Baseline (big) | Transformer | ARN (best mode) |
|---|---|---|---|
| mixed | 0.001353 | 0.001375 | **0.000284** (Discrete/PPO, early stop at 189) |
| conditional2 | 0.007223 | 0.012842 | **0.002248** (Soft, 500 epochs) |

**Key takeaways:**
- ARN outperforms baselines by **2.5–3.5×** on both tasks.
- **Dual routing modes** are essential: selection tasks require a discrete choice, while composition tasks require a mixture.
- Adaptive depth works: the model averages ~2.3 steps and dynamically shifts group weights across steps.

## 🧠 Architecture

ARN is built around a flexible computation loop.

- **Groups:** 4 parallel subnetworks with `MultiheadAttention` + FFN. They act as processing blocks.
- **Router:** Chooses the computation path.
  - **Soft mode:** Mixes all groups (differentiable, learns the mixture).
  - **Discrete mode:** Samples one group via `Categorical` (trained with PPO, learns selection).
- **Adaptive Depth:** A Readiness Evaluator decides when to stop. A PonderNet-style halt loss keeps it stable.
- **Soft Rollback:** A Progress Evaluator scores intermediate states. If a step degrades the state, it is partially rolled back (`alpha = sigmoid(...)`).
- **Budget:** A decreasing resource signal fed to the Router.
- **Stabilization:** PPO Warmup, `smooth_l1_loss`, and gradient clipping prevent training explosions.

## 📂 Project Structure

```text
arn_project/
├── arn/
│   ├── models/
│   │   ├── arn.py          # Main model loop & integration
│   │   ├── group.py        # Attention block
│   │   ├── router.py       # Soft / Discrete routing logic
│   │   └── evaluators.py   # Progress, Readiness, Value networks
│   ├── training/
│   │   ├── trainer.py      # Train loop (handles both modes, warmup, ETA)
│   │   ├── loss.py         # ARN loss (halt, aux, rollback, smooth_l1)
│   │   └── ppo.py          # GAE, PPO loss, KL early stopping
│   ├── data/
│   │   └── tasks.py        # Task generators (mixed, conditional2, etc.)
│   └── config.py           # Dataclass configs
├── scripts/
│   └── train.py            # CLI entry point
├── tests/
│   └── test_arn.py         # Unit tests
├── requirements.txt
└── README.md