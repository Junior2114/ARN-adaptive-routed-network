from dataclasses import dataclass


@dataclass
class ARNConfig:
    state_dim: int = 64
    hidden: int = 128
    n_groups: int = 4
    max_steps: int = 8
    temperature: float = 0.2
    n_heads: int = 4
    n_tokens: int = 8


@dataclass
class TrainingConfig:
    epochs: int = 200
    train_size: int = 2000
    test_size: int = 400
    lr: float = 1e-3
    lr_router: float = 3e-4
    lr_value: float = 1e-3
    aux_weight: float = 0.7
    halt_weight: float = 0.01
    balance_weight: float = 0.01
    cost_weight: float = 0.0005
    prior_lambda: float = 0.2
    gamma: float = 0.95
    lam: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.03
    target_kl: float = 0.05
    ppo_epochs: int = 3
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4
    log_every: int = 20