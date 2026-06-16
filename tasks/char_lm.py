"""
문자 레벨 언어 모델 데이터셋.

- CharDataset: 임의 텍스트 파일
- TinyShakespeare: 다운로드 or 생성 가능한 작은 텍스트 코퍼스
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class CharDataset(Dataset):
    """임의 텍스트 파일을 문자 레벨로 읽는 데이터셋."""

    def __init__(
        self,
        text: str,
        block_size: int = 128,
        stride: int = None,
    ):
        self.block_size = block_size
        stride = stride or block_size

        # 문자 → 인덱스 매핑
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        # 슬라이딩 윈도우로 청크 분할
        self.chunks: List[torch.Tensor] = []
        for i in range(0, len(data) - block_size, stride):
            self.chunks.append(data[i: i + block_size + 1])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]   # [block_size+1]

    def get_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
        )

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return ''.join(self.itos.get(int(t), '?') for t in tokens)


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

FALLBACK_TEXT = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub,
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil
Must give us pause. There's the respect
That makes calamity of so long life.
""" * 200   # ~반복해서 적당한 크기 확보


def TinyShakespeare(
    block_size: int = 128,
    cache_path: str = "/tmp/tinyshakespeare.txt",
    split: str = "train",
    train_frac: float = 0.9,
) -> CharDataset:
    """
    Tiny Shakespeare 데이터셋 로드.
    다운로드 실패 시 내장 텍스트 사용.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        try:
            print(f"Downloading TinyShakespeare from {SHAKESPEARE_URL} ...")
            with urllib.request.urlopen(SHAKESPEARE_URL, timeout=10) as r:
                text = r.read().decode("utf-8")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved to {cache_path} ({len(text):,} chars)")
        except Exception as e:
            print(f"Download failed ({e}), using fallback text.")
            text = FALLBACK_TEXT

    n = int(len(text) * train_frac)
    text_split = text[:n] if split == "train" else text[n:]
    return CharDataset(text_split, block_size=block_size)
