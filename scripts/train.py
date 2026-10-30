import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from arn.models.arn import ARN
from arn.training.trainer import train_model
from arn.utils.diagnostics import diagnose
from arn.models.baseline import Baseline, TransformerBaseline
import arn.data.tasks as data_module


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
    parser.add_argument("--router", default="soft", choices=["soft", "discrete"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_router", type=float, default=1e-4)
    parser.add_argument("--lr_value", type=float, default=1e-3)
    parser.add_argument("--aux_weight", type=float, default=0.7)
    parser.add_argument("--halt_weight", type=float, default=0.01)
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--cost_weight", type=float, default=0.0005)
    parser.add_argument("--prior_lambda", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    make_data = get_data_fn(args.task)
    x_train, y_train = make_data(args.train_size)
    x_test, y_test = make_data(args.test_size)

    print(f"task: {args.task} (train={args.train_size}, test={args.test_size}, router={args.router})")

    # Baseline'ы — только если не epochs=0
    if args.epochs > 0:
        print("=== Baseline (small) ===")
        baseline_small = Baseline(64, 64)
        train_model(baseline_small, (x_train, y_train), (x_test, y_test),
                    epochs=args.epochs, lr=args.lr, save_path="checkpoints/tmp.pt",
                    patience=args.patience, min_delta=args.min_delta)

        print("=== Baseline (big) ===")
        baseline_big = Baseline(64, 512)
        train_model(baseline_big, (x_train, y_train), (x_test, y_test),
                    epochs=args.epochs, lr=args.lr, save_path="checkpoints/tmp.pt",
                    patience=args.patience, min_delta=args.min_delta)

        print("=== Transformer ===")
        trans_baseline = TransformerBaseline(64, 128, n_heads=4, n_tokens=8, n_layers=2)
        train_model(trans_baseline, (x_train, y_train), (x_test, y_test),
                    epochs=args.epochs, lr=args.lr, save_path="checkpoints/tmp.pt",
                    patience=args.patience, min_delta=args.min_delta)

    print("=== ARN ===")
    arn = ARN(64, 128, 4, routing_mode=args.router)

    if args.epochs == 0 and args.load_path:
        # Только загрузка и оценка
        checkpoint = torch.load(args.load_path, map_location="cpu")
        arn.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded from {args.load_path} (epoch {checkpoint.get('epoch', '?')})")
        arn.eval()
        with torch.no_grad():
            pred_test, _, _, _, _, _, _, _ = arn(x_test)
            test_loss = F.mse_loss(pred_test, y_test)
        print(f"\n=== Evaluated test loss: {test_loss.item():.6f} ===")
        sys.exit(0)

    hist_arn = train_model(
        arn, (x_train, y_train), (x_test, y_test),
        epochs=args.epochs, lr=args.lr, lr_router=args.lr_router,
        lr_value=args.lr_value, aux_weight=args.aux_weight,
        halt_weight=args.halt_weight, balance_weight=args.balance_weight,
        cost_weight=args.cost_weight, prior_lambda=args.prior_lambda,
        gamma=args.gamma, weight_decay=args.weight_decay,
        patience=args.patience, min_delta=args.min_delta,
        save_path=args.save_path, load_path=args.load_path,
    )

    print("=== Итог ===")
    if hist_arn["test"]:
        print(f"ARN (last): {hist_arn['test'][-1]:.6f}")
        print(f"ARN (best): {min(hist_arn['test']):.6f}")

    print("=== Диагностика ARN ===")
    diagnose(arn, x_test, y_test)