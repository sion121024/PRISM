"""Simple character-level dataset for PRISM toy experiments."""

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 128):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.seq_len = seq_len
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.data = data

    def __len__(self):
        return max(1, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text if c in self.stoi],
                            dtype=torch.long)
