import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from arn.models.arn import ARN


def test_arn_forward_shape():
    model = ARN(64, 128, 4)
    x = torch.randn(8, 64)
    pred, traces, halt_stack, state_stack, log_probs, entropies, values = model(x)
    assert pred.shape == (8, 1)
    assert len(traces) == 8
    assert halt_stack.shape[0] == 8
    assert state_stack.shape == (8, 8, 64)
    assert log_probs.shape == (8, 8)
    assert entropies.shape == (8, 8)
    assert values.shape == (8, 8)


def test_arn_backward():
    model = ARN(64, 128, 4)
    x = torch.randn(8, 64)
    y = torch.randn(8, 1)
    pred, _, _, _, _, _, _ = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()


if __name__ == "__main__":
    test_arn_forward_shape()
    test_arn_backward()
    print("All tests passed")