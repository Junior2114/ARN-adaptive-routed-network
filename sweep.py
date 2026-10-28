import itertools
import torch

from arn import ARN, STATE_DIM, HIDDEN, N_GROUPS
from baseline import Baseline, TransformerBaseline
from train import train_model
import data as data_module


def get_data_fn(name):
    fn = getattr(data_module, f"make_{name}", None)
    if fn is None:
        raise ValueError(f"Unknown task: {name}")
    return fn


def run_one(task, cost_weight, halt_weight, balance_weight, epochs=100, seed=0):
    torch.manual_seed(seed)
    make_data = get_data_fn(task)
    x_train, y_train = make_data(1000)
    x_test, y_test = make_data(200)

    arn = ARN(STATE_DIM, HIDDEN, N_GROUPS)
    hist = train_model(
        arn, (x_train, y_train), (x_test, y_test),
        epochs=epochs, lr=1e-3,
        aux_weight=0.5,
        halt_weight=halt_weight,
        balance_weight=balance_weight,
        cost_weight=cost_weight,
        prior_lambda=0.2,
    )

    final_test = hist["test"][-1]
    final_steps = hist["steps"][-1]
    return final_test, final_steps


if __name__ == "__main__":
    tasks = ["mixed", "conditional2"]
    cost_weights = [0.0001, 0.0005, 0.001, 0.005]
    halt_weights = [0.0, 0.001, 0.01]
    balance_weight = 0.01

    print(f"{'task':<15} {'cost':<8} {'halt':<8} {'test':<10} {'steps':<8}")
    print("-" * 55)

    for task in tasks:
        for cw, hw in itertools.product(cost_weights, halt_weights):
            test_loss, steps = run_one(
                task, cw, hw, balance_weight, epochs=100
            )
            print(f"{task:<15} {cw:<8} {hw:<8} {test_loss:<10.6f} {steps:<8.2f}")