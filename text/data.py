import torch


class CharTokenizer:
    """Простой character-level токенизатор."""
    def __init__(self, text):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.idx_to_char.get(i, "?") for i in ids)


def load_shakespeare(path, max_seq_len=64, limit_chars=None):
    """Загружает текст, токенизирует и нарезает на последовательности."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if limit_chars is not None:
        text = text[:limit_chars]

    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Нарезаем на последовательности длины (max_seq_len + 1)
    n_sequences = len(ids) // (max_seq_len + 1)
    ids = ids[: n_sequences * (max_seq_len + 1)]
    ids = ids.view(n_sequences, max_seq_len + 1)

    # Вход: все токены кроме последнего
    # Target: все токены кроме первого (сдвиг на 1)
    input_ids = ids[:, :-1]
    targets = ids[:, 1:]

    return input_ids, targets, tokenizer