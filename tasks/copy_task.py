"""
Copy task: 입력 시퀀스를 그대로 출력.
M(빠른가중치)가 시퀀스 정보를 저장하는지 테스트.

포맷: [BOS, a1, a2, ..., an, SEP, a1, a2, ..., an]
BOS/SEP 이후 출력 구간에만 loss 계산.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class CopyTaskDataset(Dataset):
    BOS_IDX = 0
    SEP_IDX = 1
    DATA_OFFSET = 2  # 실제 데이터 토큰은 2부터

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 10,
        vocab_size: int = 32,     # 실제 데이터 vocab (BOS/SEP 제외)
        seed: int = 42,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.total_vocab = vocab_size + 2  # +BOS +SEP

        rng = torch.Generator().manual_seed(seed)
        # 데이터 토큰: DATA_OFFSET ~ DATA_OFFSET+vocab_size-1
        self.sequences = torch.randint(
            self.DATA_OFFSET,
            self.DATA_OFFSET + vocab_size,
            (n_samples, seq_len),
            generator=rng,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]  # [seq_len]
        bos = torch.tensor([self.BOS_IDX])
        sep = torch.tensor([self.SEP_IDX])
        full = torch.cat([bos, seq, sep, seq])  # [2*seq_len + 2]

        # loss_mask: 출력 구간만 True
        mask = torch.zeros(len(full), dtype=torch.bool)
        mask[1 + self.seq_len + 1:] = True  # SEP 이후

        return full, mask

    def get_loader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
        )

    @staticmethod
    def _collate(batch):
        tokens, masks = zip(*batch)
        return torch.stack(tokens), torch.stack(masks)


def copy_task_accuracy(
    logits: torch.Tensor,   # [B, T-1, vocab]
    tokens: torch.Tensor,   # [B, T]
    mask: torch.Tensor,     # [B, T]   (loss mask over full seq)
) -> float:
    """
    출력 구간(SEP 이후)에서 정확도 계산.
    mask: full-length, logits는 tokens[:,1:] 에 대응.
    """
    # logits position t predicts tokens[:, t+1]
    output_mask = mask[:, 1:]  # [B, T-1], aligned with logits
    if output_mask.sum() == 0:
        return 0.0

    preds = logits.argmax(-1)        # [B, T-1]
    targets = tokens[:, 1:]          # [B, T-1]
    correct = (preds == targets) & output_mask
    return correct.sum().item() / output_mask.sum().item()
