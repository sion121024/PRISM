"""
Associative Recall task: key-value 쌍 기억 후 쿼리 key로 value 검색.
PRISM의 빠른가중치 M이 연상기억으로 동작하는지 테스트.

포맷: [k1, v1, k2, v2, ..., kn, vn, SEP, query_k] → value
마지막 포지션에만 loss 계산.

토큰 인코딩:
  keys  : 0 ~ key_vocab-1
  values: key_vocab ~ key_vocab+val_vocab-1
  SEP   : key_vocab + val_vocab
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


class AssocRecallDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 10000,
        n_pairs: int = 4,
        key_vocab: int = 16,
        val_vocab: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.n_pairs = n_pairs
        self.key_vocab = key_vocab
        self.val_vocab = val_vocab
        self.SEP = key_vocab + val_vocab
        self.total_vocab = key_vocab + val_vocab + 1

        rng = torch.Generator().manual_seed(seed)
        self.data: List[Tuple[torch.Tensor, int]] = []

        for _ in range(n_samples):
            # 서로 다른 n_pairs 개의 key 선택
            keys = torch.randperm(key_vocab, generator=rng)[:n_pairs]
            vals = torch.randint(0, val_vocab, (n_pairs,), generator=rng)

            # 시퀀스: k1, v1+key_vocab, k2, v2+key_vocab, ...
            pairs = torch.stack([keys, vals + key_vocab], dim=1).flatten()

            # 쿼리: pairs 중 하나
            q_idx = torch.randint(0, n_pairs, (1,), generator=rng).item()
            query_k = keys[q_idx].unsqueeze(0)
            answer_v = int(vals[q_idx].item())

            sep = torch.tensor([self.SEP])
            seq = torch.cat([pairs, sep, query_k])
            self.data.append((seq, answer_v))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx]

    def get_loader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
        )

    @staticmethod
    def _collate(batch):
        seqs, labels = zip(*batch)
        return torch.stack(seqs), torch.tensor(labels, dtype=torch.long)


class AssocRecallModel(torch.nn.Module):
    """
    PRISM 연상기억 래퍼.
    input: [k1,v1,...,kn,vn,SEP,query_k]
    output: value logit for last position.
    """

    def __init__(self, prism_model, val_vocab_offset: int):
        super().__init__()
        self.model = prism_model
        self.val_vocab_offset = val_vocab_offset

    def forward(
        self, tokens: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """
        tokens: [B, T]
        labels: [B] — value indices (0-based, without offset)
        """
        import torch.nn.functional as F

        B, T = tokens.shape
        device = tokens.device

        x, mem = self.model.init_state(B, device)
        for t in range(T - 1):   # process all but last input token
            u = self.model.embed(tokens[:, t])
            x, mem = self.model.cell(u, mem, x)

        # 마지막 토큰 처리 → value 예측
        u_last = self.model.embed(tokens[:, -1])
        x, mem = self.model.cell(u_last, mem, x)
        logits_full = self.model.output_proj(x)  # [B, total_vocab]

        # value 부분만 추출: [val_vocab_offset : val_vocab_offset + val_vocab]
        offset = self.val_vocab_offset
        val_vocab = logits_full.shape[-1] - offset - 1  # exclude SEP
        logits_val = logits_full[:, offset:offset + val_vocab]

        loss = F.cross_entropy(logits_val, labels)
        acc = (logits_val.argmax(-1) == labels).float().mean().item()

        return {'loss': loss, 'acc': acc, 'logits': logits_val}
