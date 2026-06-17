"""
멀티모달 학습 검증: 시각 항이 실제로 학습에 기여하는가.

태스크: 숫자 캡션 (Digit Captioning)
  - 이미지: 28×28 합성 숫자 패턴
  - 텍스트: 숫자를 설명하는 짧은 문장 ("zero", "one", ...)
  - 멀티모달 모델이 텍스트 전용 대비 낮은 ppl 달성하면 시각 항 효과 입증

실행:
  python verify_multimodal.py
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)

from prism import PRISMMultimodalModel, PRISMLangModel


# ------------------------------------------------------------------ #
# 합성 데이터: 숫자 이미지 + 텍스트 캡션                              #
# ------------------------------------------------------------------ #

DIGIT_WORDS = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]

def build_vocab():
    chars = sorted(set(" ".join(DIGIT_WORDS)))
    stoi = {c: i+1 for i, c in enumerate(chars)}
    stoi["<pad>"] = 0
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos

STOI, ITOS = build_vocab()
VOCAB_SIZE = len(STOI)


def make_digit_image(digit: int, size: int = 28) -> torch.Tensor:
    """숫자를 나타내는 간단한 패턴 이미지 (훈련 신호용)."""
    img = torch.zeros(1, size, size)
    # 숫자별 고유 패턴: 대각선 + 숫자에 따른 강도 변화
    for i in range(size):
        for j in range(size):
            val = math.sin(i * (digit + 1) * 0.4) * math.cos(j * (digit + 1) * 0.3)
            img[0, i, j] = (val + 1) / 2
    return img


class DigitCaptionDataset(Dataset):
    def __init__(self, n_samples: int = 2000, seq_len: int = 6, seed: int = 42):
        torch.manual_seed(seed)
        self.data = []
        for _ in range(n_samples):
            digit = torch.randint(0, 10, ()).item()
            word = DIGIT_WORDS[digit]
            # 텍스트: "<digit_word>" 패딩
            toks = [STOI[c] for c in word]
            toks = toks[:seq_len]
            toks += [0] * (seq_len - len(toks))
            tokens = torch.tensor(toks, dtype=torch.long)
            image = make_digit_image(digit)
            self.data.append((tokens, image, digit))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate(batch):
    tokens = torch.stack([b[0] for b in batch])
    images = torch.stack([b[1] for b in batch])
    return tokens, images


def train_eval(model, loader, val_loader, n_epochs, name, use_images=True):
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_ppl = float("inf")
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for tokens, images in loader:
            imgs = images.unsqueeze(1).expand(-1, tokens.shape[1], -1, -1, -1) \
                   if use_images else None
            out = model(tokens, imgs)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += out["loss"].item()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for tokens, images in val_loader:
                imgs = images.unsqueeze(1).expand(-1, tokens.shape[1], -1, -1, -1) \
                       if use_images else None
                out = model(tokens, imgs)
                vloss += out["loss"].item()
        vppl = math.exp(vloss / len(val_loader))
        best_ppl = min(best_ppl, vppl)
        print(f"  [{name}] epoch {epoch} | val_ppl {vppl:.3f}")
    return best_ppl


def main():
    print("=" * 52)
    print("멀티모달 학습 검증: 숫자 캡션 태스크")
    print("=" * 52)
    print(f"vocab={VOCAB_SIZE} | digits=10 | img=28×28")

    train_ds = DigitCaptionDataset(n_samples=2000, seq_len=6, seed=0)
    val_ds   = DigitCaptionDataset(n_samples=400,  seq_len=6, seed=99)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate)

    N_EPOCHS = 5

    # A: 멀티모달 (텍스트 + 이미지)
    print("\n--- A: 멀티모달 (텍스트 + 이미지) ---")
    mm_model = PRISMMultimodalModel(
        vocab_size=VOCAB_SIZE, d=128, emb_dim=32, K=4,
        mem_rank=8, dec_hidden=32, vis_dim=32,
        img_size=28, patch_size=7, in_channels=1,
        use_prior=True,
    )
    print(f"params: {mm_model.num_params():,}")
    ppl_mm = train_eval(mm_model, train_loader, val_loader, N_EPOCHS,
                        "Multimodal", use_images=True)

    # B: 텍스트 전용 (동일 모델, 이미지 없음)
    print("\n--- B: 텍스트 전용 (이미지 없음) ---")
    txt_model = PRISMMultimodalModel(
        vocab_size=VOCAB_SIZE, d=128, emb_dim=32, K=4,
        mem_rank=8, dec_hidden=32, vis_dim=32,
        img_size=28, patch_size=7, in_channels=1,
        use_prior=True,
    )
    print(f"params: {txt_model.num_params():,}")
    ppl_txt = train_eval(txt_model, train_loader, val_loader, N_EPOCHS,
                         "TextOnly", use_images=False)

    print("\n" + "=" * 52)
    print(f"  Multimodal (텍스트+이미지): {ppl_mm:.3f} ppl")
    print(f"  TextOnly   (텍스트만):      {ppl_txt:.3f} ppl")
    delta = ppl_txt - ppl_mm
    if delta > 0:
        print(f"\n  시각 항 효과: +{delta:.3f} ppl 개선 ← 이미지가 실제로 도움")
    else:
        print(f"\n  시각 항 효과: {delta:.3f} (이미지 미기여 — 태스크 재설계 필요)")


if __name__ == "__main__":
    main()
