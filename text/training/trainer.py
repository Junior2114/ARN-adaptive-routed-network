import time
import torch
import torch.nn.functional as F

from text.models.arn import TextARN
from text.training.loss import text_arn_loss


def train_text_model(model, train_data, test_data, epochs=20, lr=1e-3,
                     batch_size=32, aux_weight=0.5, halt_weight=0.01,
                     prior_lambda=0.2, log_every=1):
    device = next(model.parameters()).device
    input_train, target_train = train_data
    input_test, target_test = test_data

    n_train = input_train.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "test": []}

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        perm = torch.randperm(n_train)
        input_train_shuffled = input_train[perm]
        target_train_shuffled = target_train[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            input_batch = input_train_shuffled[i:i+batch_size].to(device)
            target_batch = target_train_shuffled[i:i+batch_size].to(device)

            logits, traces, halt_stack, state_stack = model(input_batch)
            loss, main_loss, aux_loss, halt_loss = text_arn_loss(
                model, logits, target_batch, halt_stack, traces, state_stack,
                aux_weight=aux_weight, halt_weight=halt_weight,
                prior_lambda=prior_lambda,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        model.eval()
        total_test = 0.0
        n_test_batches = 0
        total_steps = 0.0

        with torch.no_grad():
            for i in range(0, input_test.shape[0], batch_size):
                input_batch = input_test[i:i+batch_size].to(device)
                target_batch = target_test[i:i+batch_size].to(device)

                logits_test, _, halt_stack_test, _ = model(input_batch)
                test_loss = F.cross_entropy(
                    logits_test.reshape(-1, logits_test.size(-1)),
                    target_batch.reshape(-1)
                )
                total_test += test_loss.item()
                n_test_batches += 1

                steps_tensor = torch.arange(
                    halt_stack_test.size(0),
                    device=halt_stack_test.device
                ).float().unsqueeze(1)
                total_steps += (steps_tensor * halt_stack_test).sum(dim=0).mean().item()

        test_loss = total_test / n_test_batches
        steps_val = total_steps / n_test_batches

        history["train"].append(train_loss)
        history["test"].append(test_loss)

        epoch_time = time.time() - epoch_start
        eta = (time.time() - start_time) / (epoch + 1) * (epochs - epoch - 1)

        if epoch % log_every == 0:
            print(f"epoch={epoch:3d} train={train_loss:.4f} "
                  f"test={test_loss:.4f} steps={steps_val:.2f} "
                  f"| {epoch_time:.1f}s (ETA: {eta/60:.1f}m)")

    print(f"\nTotal time: {time.time() - start_time:.2f}s")
    return history