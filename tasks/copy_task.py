"""
Associative recall / copy task — canonical test for memory architectures.

Task: given  [k1 v1 k2 v2 ... kN vN SEP k_query]
       predict v_query

This is where energy-based fast weights (M) should shine vs vanilla RNN:
the model must store (key→value) associations in M at test time.
"""

import torch
from torch.utils.data import Dataset


class CopyTaskDataset(Dataset):
    """
    Copy task: sequence → same sequence (shift by 1).
    Simplest possible memory test.
    """

    def __init__(self, vocab_size=16, seq_len=32, n_samples=4000, seed=0):
        torch.manual_seed(seed)
        self.data = torch.randint(2, vocab_size, (n_samples, seq_len + 1))
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


class AssociativeRecallDataset(Dataset):
    """
    Associative recall: [k1 v1 k2 v2 ... kN vN SEP ki] -> vi

    At the SEP token the model must recall which value was paired with ki.
    Tests whether fast-weight M actually stores key-value associations.
    """

    def __init__(self, vocab_size=16, n_pairs=4, n_samples=4000, seed=0):
        torch.manual_seed(seed)
        self.n_pairs   = n_pairs
        self.vocab_size = vocab_size
        # token 0 = SEP, tokens 1..vocab_size-1 = keys/values
        seqs, targets = [], []
        for _ in range(n_samples):
            keys   = torch.randperm(vocab_size - 1)[:n_pairs] + 1
            values = torch.randperm(vocab_size - 1)[:n_pairs] + 1
            pairs  = torch.stack([keys, values], dim=1).flatten()  # k1v1k2v2...
            sep    = torch.tensor([0])
            query_idx = torch.randint(0, n_pairs, (1,)).item()
            query  = keys[query_idx:query_idx+1]
            target = values[query_idx].item()
            seq    = torch.cat([pairs, sep, query])
            seqs.append(seq)
            targets.append(target)
        self.seqs    = torch.stack(seqs)
        self.targets = torch.tensor(targets)
        self.seq_len = self.seqs.shape[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]
