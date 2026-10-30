# ARN — Adaptive Routed Network

Adaptive computation with attention groups, soft rollback, budget, and PPO-trained discrete router. Includes both vector-based and text-based implementations.

## Results

### Vector Tasks (1.18M params, train=5000)

| Task | Baseline (big) | Transformer | ARN (best mode) |
|---|---|---|---|
| mixed | 0.001353 | 0.001375 | **0.000284** (Discrete/PPO, early stop at 189) |
| conditional2 | 0.005400 | 0.006500 | **0.002248** (Soft, 500 epochs) |

### Text Task (Tiny Shakespeare, 1.18M params)

| Model | Vocab | Params | Test Loss | Avg Steps |
|---|---|---|---|---|
| ARN (Text) | 65 (char-level) | 1.18M | **0.0364** | 2.3 |

**Generation example:**
> `ROMEO: witend to a tee ce to ournterest onght you dong thee: hill, What rall to this he witentun you soght seeds mast not the he feoDs hat not`

## Architecture

### Core ARN (Vector)
- **Groups:** Attention over tokens (`MultiheadAttention` + FFN)
- **Router:** Discrete (`Categorical`, PPO) or Soft (mixture)
- **Soft rollback:** `alpha = sigmoid((progress - best) / temperature)`
- **Readiness:** `p_stop` for adaptive depth
- **Budget:** Decreases per step, visible to Router
- **PPO:** GAE, advantage normalization, clipped objective, KL early stopping

### Text ARN
- **Embedding:** `Embedding(vocab, dim) + PositionalEmbedding`
- **Groups:** `TransformerEncoderLayer` (batch_first=True)
- **Router:** Soft mixture (for stability on text)
- **Output:** `Linear(dim, vocab_size)`
- **Loss:** Cross-entropy per token

## Install

```bash
pip install -r requirements.txt