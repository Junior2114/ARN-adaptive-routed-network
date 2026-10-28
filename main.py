import argparse
import torch

from arn import ARN, STATE_DIM, HIDDEN, N_GROUPS
from baseline import Baseline, TransformerBaseline
from train import train_model, diagnose

import data as data_module


def get_data_fn(name):
    fn = getattr(data_module, f"make_{name}", None)
    if fn is None:
        raise ValueError(f"Unknown task: {name}")
    return fn


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mixed")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_router", type=float, default=3e-4)
    parser.add_argument("--lr_value", type=float, default=1e-3)
    parser.add_argument("--aux_weight", type=float, default=0.7)
    parser.add_argument("--halt_weight", type=float, default=0.01)
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--cost_weight", type=float, default=0.0005)
    parser.add_argument("--prior_lambda", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--baseline_small_hidden", type=int, default=64)
    parser.add_argument("--baseline_big_hidden", type=int, default=512)
    parser.add_argument("--trans_hidden", type=int, default=128)
    parser.add_argument("--trans_layers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    make_data = get_data_fn(args.task)
    x_train, y_train = make_data(args.train_size)
    x_test, y_test = make_data(args.test_size)

    print(f"task: {args.task} (train={args.train_size}, test={args.test_size})")

    print("=== Baseline (small) ===")
    baseline_small = Baseline(STATE_DIM, args.baseline_small_hidden)
    print(f"params: {count_params(baseline_small)}")
    hist_base_small = train_model(baseline_small, (x_train, y_train), (x_test, y_test), epochs=args.epochs, lr=args.lr)

    print("=== Baseline (big) ===")
    baseline_big = Baseline(STATE_DIM, args.baseline_big_hidden)
    print(f"params: {count_params(baseline_big)}")
    hist_base_big = train_model(baseline_big, (x_train, y_train), (x_test, y_test), epochs=args.epochs, lr=args.lr)

    print("=== Transformer ===")
    trans_baseline = TransformerBaseline(STATE_DIM, args.trans_hidden, n_heads=4, n_tokens=8, n_layers=args.trans_layers)
    print(f"params: {count_params(trans_baseline)}")
    hist_trans = train_model(trans_baseline, (x_train, y_train), (x_test, y_test), epochs=args.epochs, lr=args.lr)

    print("=== ARN (PPO) ===")
    arn = ARN(STATE_DIM, HIDDEN, N_GROUPS)
    print(f"params: {count_params(arn)}")
    hist_arn = train_model(
        arn, (x_train, y_train), (x_test, y_test),
        epochs=args.epochs, lr=args.lr, lr_router=args.lr_router,
        lr_value=args.lr_value,
        aux_weight=args.aux_weight, halt_weight=args.halt_weight,
        balance_weight=args.balance_weight, cost_weight=args.cost_weight,
        prior_lambda=args.prior_lambda,
        gamma=args.gamma, weight_decay=args.weight_decay,
    )

    print("=== Итог ===")
    print(f"Baseline (small): {hist_base_small['test'][-1]:.6f}")
    print(f"Baseline (big):   {hist_base_big['test'][-1]:.6f}")
    print(f"Transformer:      {hist_trans['test'][-1]:.6f}")
    print(f"ARN:              {hist_arn['test'][-1]:.6f}")

    print("=== Диагностика ARN ===")
    diagnose(arn, x_test, y_test)