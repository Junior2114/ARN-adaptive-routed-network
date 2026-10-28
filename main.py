import argparse
import torch

from arn import ARN, STATE_DIM, HIDDEN, N_GROUPS
from baseline import Baseline, TransformerBaseline
from train import train_model, diagnose

import data as data_module


def get_data_fn(name):
    fn = getattr(data_module, f"make_{name}", None)
    if fn is None:
        raise ValueError(f"Unknown task: {name}. "
                         f"Available: mixed, sqrt, conditional, conditional2, "
                         f"sequential_sum, sort")
    return fn


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mixed",
                        help="Task name: mixed, sqrt, conditional, conditional2, "
                             "sequential_sum, sort")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux_weight", type=float, default=0.5)
    parser.add_argument("--halt_weight", type=float, default=0.01)
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--cost_weight", type=float, default=0.0005)
    parser.add_argument("--prior_lambda", type=float, default=0.2)
    parser.add_argument("--baseline_small_hidden", type=int, default=64,
                        help="Hidden size for small Baseline")
    parser.add_argument("--baseline_big_hidden", type=int, default=512,
                        help="Hidden size for big Baseline (match ARN)")
    parser.add_argument("--trans_hidden", type=int, default=128,
                        help="FFN hidden size for TransformerBaseline")
    parser.add_argument("--trans_layers", type=int, default=2,
                        help="Number of transformer layers")
    args = parser.parse_args()

    torch.manual_seed(0)

    make_data = get_data_fn(args.task)
    x_train, y_train = make_data(1000)
    x_test, y_test = make_data(200)

    print(f"task: {args.task}")
    print(f"train: {x_train.shape}, {y_train.shape}")
    print(f"test:  {x_test.shape}, {y_test.shape}")
    print()

    # Baseline (small)
    print("=== Baseline (small) ===")
    baseline_small = Baseline(STATE_DIM, args.baseline_small_hidden)
    print(f"params: {count_params(baseline_small)}")
    hist_base_small = train_model(
        baseline_small, (x_train, y_train), (x_test, y_test),
        epochs=args.epochs,
        lr=args.lr,
    )

    # Baseline (big)
    print()
    print("=== Baseline (big) ===")
    baseline_big = Baseline(STATE_DIM, args.baseline_big_hidden)
    print(f"params: {count_params(baseline_big)}")
    hist_base_big = train_model(
        baseline_big, (x_train, y_train), (x_test, y_test),
        epochs=args.epochs,
        lr=args.lr,
    )

    # Transformer Baseline
    print()
    print("=== Transformer Baseline ===")
    trans_baseline = TransformerBaseline(
        STATE_DIM, args.trans_hidden,
        n_heads=4, n_tokens=8, n_layers=args.trans_layers,
    )
    print(f"params: {count_params(trans_baseline)}")
    hist_trans = train_model(
        trans_baseline, (x_train, y_train), (x_test, y_test),
        epochs=args.epochs,
        lr=args.lr,
    )

    # ARN
    print()
    print("=== ARN (PonderNet-style) ===")
    arn = ARN(STATE_DIM, HIDDEN, N_GROUPS)
    print(f"params: {count_params(arn)}")
    hist_arn = train_model(
        arn, (x_train, y_train), (x_test, y_test),
        epochs=args.epochs,
        lr=args.lr,
        aux_weight=args.aux_weight,
        halt_weight=args.halt_weight,
        balance_weight=args.balance_weight,
        cost_weight=args.cost_weight,
        prior_lambda=args.prior_lambda,
    )

    print()
    print("=== Итог ===")
    print(f"Baseline (small):   {hist_base_small['test'][-1]:.6f}  "
          f"(params: {count_params(baseline_small)})")
    print(f"Baseline (big):     {hist_base_big['test'][-1]:.6f}  "
          f"(params: {count_params(baseline_big)})")
    print(f"Transformer:        {hist_trans['test'][-1]:.6f}  "
          f"(params: {count_params(trans_baseline)})")
    print(f"ARN:                {hist_arn['test'][-1]:.6f}  "
          f"(params: {count_params(arn)})")

    print()
    print("=== Диагностика ARN ===")
    diagnose(arn, x_test, y_test)