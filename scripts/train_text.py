import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F

from text.config import TextARNConfig
from text.data import load_shakespeare
from text.models.arn import TextARN
from text.training.trainer import train_text_model


def generate_text(model, tokenizer, prompt, max_new_tokens=200, max_seq_len=64):
    device = next(model.parameters()).device
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_trimmed = input_ids[:, -max_seq_len:]
            logits, _, _, _ = model(input_trimmed)
            next_logits = logits[0, -1]
            probs = F.softmax(next_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit_chars", type=int, default=None)
    parser.add_argument("--data_path", type=str, default="text/data/shakespeare.txt")
    parser.add_argument("--save_path", type=str, default="checkpoints/arn-text.pt")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Путь к чекпоинту для дообучения")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.data_path}...")
    input_ids, targets, tokenizer = load_shakespeare(
        args.data_path, args.max_seq_len, args.limit_chars
    )
    print(f"Total sequences: {input_ids.shape[0]}, Tokens: {input_ids.shape[1]}, Vocab: {tokenizer.vocab_size}")

    split = int(0.9 * input_ids.shape[0])
    train_data = (input_ids[:split], targets[:split])
    test_data = (input_ids[split:], targets[split:])
    print(f"Train: {train_data[0].shape[0]}, Test: {test_data[0].shape[0]}")

    cfg = TextARNConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
    )
    model = TextARN(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        dim=cfg.dim,
        hidden=cfg.hidden,
        n_groups=cfg.n_groups,
        max_steps=cfg.max_steps,
        temperature=cfg.temperature,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Загрузка чекпоинта (если указан)
    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded. Resuming training...")

    hist = train_text_model(
        model,
        train_data,
        test_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    print(f"\nFinal test loss: {hist['test'][-1]:.4f}")

    print("\n=== Generation ===")
    print(generate_text(model, tokenizer, "ROMEO: ", 200, args.max_seq_len))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "tokenizer": {
            "char_to_idx": tokenizer.char_to_idx,
            "idx_to_char": tokenizer.idx_to_char,
        }
    }, args.save_path)
    print(f"\nModel saved to {args.save_path}")