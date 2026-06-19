"""
Stage 4 (언어): PRISM 한국어 + 영어 바이트 단위 언어 모델.

PRISM은 바이트 단위(vocab 256)라 UTF-8로 한국어·영어를 토크나이저 없이
같은 고정크기 상태 x에 압축해 학습한다 (설계철학: 모든 입력 = 같은 E 하강).

데이터 (Kaggle 마운트):
  - 한국어: junbumlee/kcbert-... (뉴스 댓글, UTF-8 텍스트)
  - 영어:   nguyenatu/enwik8 (enwik8-train.txt)

지표: 언어별 val bits-per-char(byte). 생성: 한국어·영어 프롬프트 각각.

실행:
  python stage22_bilingual.py --smoke
  python stage22_bilingual.py --device cuda --d 512 --ko_mb 12 --en_mb 12
"""

import argparse
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from prism import PRISMLangModel


def find_largest(root_substrs, exts=(".txt",)):
    """/kaggle/input 아래에서 경로에 substr 포함 + 확장자 일치하는 최대 파일."""
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    best, best_sz = None, -1
    for dp, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dp, f)
            low = full.lower()
            if any(s in low for s in root_substrs) and low.endswith(exts):
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    continue
                if sz > best_sz:
                    best, best_sz = full, sz
    return best


def read_bytes(path, n, skip=0):
    with open(path, "rb") as f:
        if skip:
            f.seek(skip)
        return f.read(n)


def load_korean(n_bytes):
    p = find_largest(["kcbert", "korean", "namu"], (".txt",))
    if p:
        print(f"Korean: {p}", flush=True)
        # 앞부분 헤더/중복 회피 위해 약간 건너뜀
        return read_bytes(p, n_bytes, skip=1_000_000)
    print("Korean source not mounted; using tiny fallback.", flush=True)
    return ("안녕하세요. 한국어 언어 모델입니다. 오늘 날씨가 좋네요. " * 2000).encode("utf-8")[:n_bytes]


def load_english(n_bytes):
    p = find_largest(["enwik8-train", "enwik8_train"], (".txt",))
    if not p:
        p = find_largest(["enwik8"], (".txt",))
    if p:
        print(f"English: {p}", flush=True)
        return read_bytes(p, n_bytes, skip=1_000_000)
    try:
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
               "data/tinyshakespeare/input.txt")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = r.read()
        return (d * (n_bytes // len(d) + 1))[:n_bytes]
    except Exception:
        return ("The quick brown fox jumps over the lazy dog. " * 5000).encode()[:n_bytes]


class ByteDataset(Dataset):
    def __init__(self, data: bytes, block_size: int, stride: int = None):
        stride = stride or block_size
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self.chunks = [t[i:i + block_size + 1]
                       for i in range(0, len(t) - block_size - 1, stride)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]


def evaluate(model, loader, device):
    model.eval()
    tot, nb = 0.0, 0
    with torch.no_grad():
        for tokens in loader:
            tokens = tokens.to(device)
            tot += model(tokens)["loss"].item(); nb += 1
    loss = tot / max(nb, 1)
    return loss / math.log(2)   # bpc


def sample(model, seed_text, device, n=160, temperature=0.8, top_k=40):
    model.eval()
    b = seed_text.encode("utf-8")
    prompt = torch.tensor([list(b)], dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_tokens=n, temperature=temperature, top_k=top_k)
    gen = bytes(out[0].tolist())
    return gen.decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--block_size", type=int, default=192)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--mem_rank", type=int, default=48)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--tbptt", type=int, default=48)
    p.add_argument("--ko_mb", type=int, default=12, help="한국어 MB")
    p.add_argument("--en_mb", type=int, default=12, help="영어 MB")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.device = "cpu"; args.d = 64; args.emb_dim = 32; args.K = 2
        args.n_layers = 1; args.mem_rank = 8; args.block_size = 64
        args.batch_size = 16; args.epochs = 1; args.ko_mb = 0; args.en_mb = 0
        args.tbptt = 16; args.max_steps = 5; args.amp = False

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device(args.device)

    print("=" * 64)
    print("Stage 4 (언어): PRISM 한국어 + 영어 바이트 LM")
    print("=" * 64)
    ko_n = (args.ko_mb or 1) * 1_000_000
    en_n = (args.en_mb or 1) * 1_000_000
    ko = load_korean(ko_n)
    en = load_english(en_n)
    print(f"한국어 {len(ko):,}B | 영어 {len(en):,}B | device={device}", flush=True)

    # train/val split per language, then concat for training
    def split(data, frac=0.97):
        n = int(len(data) * frac); return data[:n], data[n:]
    ko_tr, ko_va = split(ko); en_tr, en_va = split(en)
    train_data = ko_tr + en_tr     # 두 언어를 같은 바이트 스트림에 결합

    train_ds = ByteDataset(train_data, args.block_size)
    ko_val = ByteDataset(ko_va, args.block_size)
    en_val = ByteDataset(en_va, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    ko_loader = DataLoader(ko_val, batch_size=args.batch_size)
    en_loader = DataLoader(en_val, batch_size=args.batch_size)
    print(f"train_chunks={len(train_ds):,} | ko_val={len(ko_val):,} | en_val={len(en_val):,}",
          flush=True)

    model = PRISMLangModel(
        vocab_size=256, d=args.d, emb_dim=args.emb_dim, K=args.K,
        alpha=0.05, lam=0.01, memory_mode="sliding",
        mem_rank=args.mem_rank, mem_scale=4.0,
        decoder="mlp", dec_hidden=args.d, simple_prior=True, use_urec=True,
        use_gate=True, use_conv=True, d_conv=4,
        approximate_grad=True, n_layers=args.n_layers,
    ).to(device)
    print(f"PRISM params: {model.num_params():,}", flush=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    use_amp = (device.type == "cuda" and args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t_start = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); ep_loss, nb, t0 = 0.0, 0, time.time()
        for tokens in train_loader:
            tokens = tokens.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(tokens, tbptt_window=args.tbptt)["loss"]
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item(); nb += 1
            if args.max_steps and nb >= args.max_steps:
                break
        ko_bpc = evaluate(model, ko_loader, device)
        en_bpc = evaluate(model, en_loader, device)
        mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
        print(f"ep {epoch:2d} | train_loss {ep_loss/nb:.4f} | "
              f"한국어 bpc {ko_bpc:.4f} | 영어 bpc {en_bpc:.4f} | "
              f"{time.time()-t0:.0f}s | {mem:.1f}GB", flush=True)

        # 매 epoch 생성 샘플
        print("  [한국어] " + sample(model, "오늘", device, n=120).replace("\n", " ⏎ "), flush=True)
        print("  [English] " + sample(model, "The ", device, n=120).replace("\n", " ⏎ "), flush=True)

    print("\n" + "=" * 64)
    print("최종 생성 샘플:")
    for seed in ["한국", "나는", "오늘 날씨는"]:
        print(f"  [KO '{seed}'] " + sample(model, seed, device, n=200).replace("\n", " ⏎ "))
    for seed in ["The ", "In the ", "I think "]:
        print(f"  [EN '{seed}'] " + sample(model, seed, device, n=200).replace("\n", " ⏎ "))


if __name__ == "__main__":
    main()
