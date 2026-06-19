"""
Stage 4: GPU 스케일업 — PRISM vs Mamba (enwik8 char-LM).

de-risking Stage 4: "V100/GPU에서 50~150M 스케일업".
CPU 실험(~100K params)에서 GPU 대형 모델로 100배+ 스케일업하여
PRISM이 (1) 안정적으로 학습되고 (2) 파라미터 매칭 Mamba와 경쟁하는지 검증.

설계철학 유지:
  - 느린가중치 θ: K-step unroll backprop (approximate_grad로 그래프 1/K 축소)
  - 빠른가중치 M: Hebbian (gradient detach)
  - 이중시계: 외부 t(토큰) × 내부 s(K-step 에너지 하강)

데이터: enwik8 (표준 char-LM 벤치마크, 바이트 단위). 다운로드 실패 시 TinyShakespeare.
지표: bits-per-character (bpc) = val_loss / ln(2).

실행:
  python stage20_gpu_scaleup.py --smoke              # 로컬 CPU 작동 확인
  python stage20_gpu_scaleup.py --device cuda --d 512 --n_layers 2 --epochs 3
"""

import argparse
import math
import os
import time
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from prism import PRISMLangModel
from baselines.mamba_lm import MambaBlock


# ------------------------------------------------------------------ #
# 데이터: enwik8 (subset) char-level                                  #
# ------------------------------------------------------------------ #

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def _find_kaggle_file(*needles):
    """/kaggle/input 아래에서 이름에 needle을 포함하는 파일 경로 반환."""
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    for dp, _, files in os.walk(root):
        for f in files:
            low = f.lower()
            if any(n in low for n in needles):
                return os.path.join(dp, f)
    return None


def load_enwik8_split(n_train_chars: int, n_val_chars: int):
    """
    (train_bytes, val_bytes) 반환.

    우선순위:
      1. Kaggle 마운트 데이터셋 (/kaggle/input/.../enwik8-train.txt, -valid.txt)
      2. mattmahoney enwik8.zip 다운로드 (커널 internet)
      3. TinyShakespeare (github raw) fallback
    """
    tr = _find_kaggle_file("enwik8-train", "enwik8_train")
    va = _find_kaggle_file("enwik8-valid", "enwik8-test", "enwik8_valid")
    if tr and va:
        print(f"Using mounted enwik8: {tr} | {va}", flush=True)
        with open(tr, "rb") as f:
            train = f.read(n_train_chars)
        with open(va, "rb") as f:
            val = f.read(n_val_chars)
        return train, val

    # 다운로드 시도
    try:
        print(f"Downloading enwik8 from {ENWIK8_URL} ...", flush=True)
        zpath = "/tmp/enwik8.zip"
        req = urllib.request.Request(ENWIK8_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(zpath, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zpath) as z:
            data = z.read("enwik8")
        print(f"  enwik8: {len(data):,} bytes", flush=True)
        return data[:n_train_chars], data[n_train_chars:n_train_chars + n_val_chars]
    except Exception as e:
        print(f"enwik8 unavailable ({e}); falling back to TinyShakespeare.", flush=True)
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
               "data/tinyshakespeare/input.txt")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            data = data * (max(1, (n_train_chars + n_val_chars) // len(data) + 1))
        except Exception:
            from tasks.char_lm import FALLBACK_TEXT
            data = (FALLBACK_TEXT * 50).encode()
        n = int(len(data) * 0.95)
        return data[:min(n, n_train_chars)], data[n:n + n_val_chars]


class ByteDataset(Dataset):
    def __init__(self, data: bytes, block_size: int, stride: int = None):
        stride = stride or block_size
        self.block_size = block_size
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self.chunks = [t[i:i + block_size + 1]
                       for i in range(0, len(t) - block_size - 1, stride)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]


# ------------------------------------------------------------------ #
# 베이스라인: 다층 Mamba (파라미터 매칭용)                            #
# ------------------------------------------------------------------ #

class MambaStack(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=2, d_state=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        for b in self.blocks:
            nn.init.constant_(b.dt_proj.bias, math.log(math.expm1(1.0)))

    def forward(self, tokens):
        x = self.embed(tokens[:, :-1])
        for b in self.blocks:
            x, _ = b(x)
        x = self.norm_f(x)
        logits = self.output_proj(x)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                               tokens[:, 1:].reshape(-1))
        return {"loss": loss, "logits": logits}

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------ #
# 학습/평가                                                            #
# ------------------------------------------------------------------ #

def evaluate(model, loader, device, is_prism):
    model.eval()
    tot, nb = 0.0, 0
    with torch.no_grad():
        for tokens in loader:
            tokens = tokens.to(device)
            out = model(tokens)
            tot += out["loss"].item()
            nb += 1
    val_loss = tot / max(nb, 1)
    return val_loss, val_loss / math.log(2)   # (loss, bpc)


def train_model(model, name, train_loader, val_loader, args, device, is_prism):
    model = model.to(device)
    n = model.num_params()
    print(f"\n=== {name}: {n:,} params ===", flush=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    use_amp = (device.type == "cuda" and args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_bpc = float("inf")
    tok_seen, t_start = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb, t0 = 0.0, 0, time.time()
        for tokens in train_loader:
            tokens = tokens.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = (model(tokens, tbptt_window=args.tbptt)
                       if is_prism else model(tokens))
                loss = out["loss"]
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item(); nb += 1
            tok_seen += tokens.numel()
            if args.max_steps and nb >= args.max_steps:
                break
        val_loss, bpc = evaluate(model, val_loader, device, is_prism)
        best_bpc = min(best_bpc, bpc)
        dt = time.time() - t0
        tps = tok_seen / (time.time() - t_start)
        mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
        print(f"  ep {epoch:2d} | train_loss {ep_loss/nb:.4f} | "
              f"val_bpc {bpc:.4f} | {dt:.0f}s | {tps:,.0f} tok/s | {mem:.1f}GB",
              flush=True)
    return best_bpc, n


def build_prism(vocab, args):
    return PRISMLangModel(
        vocab_size=vocab, d=args.d, emb_dim=args.emb_dim, K=args.K,
        alpha=0.05, lam=0.01,
        memory_mode="sliding",          # 저랭크 Hebbian — 대형 d에서 메모리 효율적
        mem_rank=args.mem_rank, mem_scale=4.0,
        decoder="mlp", dec_hidden=args.d,
        simple_prior=True, use_urec=True,
        use_gate=True, use_conv=True, d_conv=4,
        approximate_grad=args.approximate_grad,
        n_layers=args.n_layers,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--mem_rank", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--approximate_grad", action="store_true", default=True)
    p.add_argument("--tbptt", type=int, default=64, help="PRISM truncated BPTT 윈도우")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--n_chars", type=int, default=20_000_000)
    p.add_argument("--n_val_chars", type=int, default=2_000_000)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--mamba_d", type=int, default=0, help="0=PRISM의 d와 동일")
    p.add_argument("--skip_mamba", action="store_true")
    p.add_argument("--skip_prism", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.device = "cpu"; args.d = 64; args.emb_dim = 32; args.K = 2
        args.n_layers = 1; args.mem_rank = 8; args.block_size = 64
        args.batch_size = 16; args.epochs = 1; args.n_chars = 60_000
        args.n_val_chars = 10_000; args.max_steps = 5; args.amp = False
        args.tbptt = 16

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 미가용 → CPU", flush=True); args.device = "cpu"
    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device(args.device)

    print("=" * 64)
    print("Stage 4: GPU 스케일업 — PRISM vs Mamba (enwik8 char-LM)")
    print("=" * 64)
    print(f"device={device} | block={args.block_size} | batch={args.batch_size} | "
          f"K={args.K} | n_layers={args.n_layers} | amp={args.amp}", flush=True)

    train_bytes, val_bytes = load_enwik8_split(args.n_chars, args.n_val_chars)
    vocab = 256
    train_ds = ByteDataset(train_bytes, args.block_size)
    val_ds = ByteDataset(val_bytes, args.block_size)
    data = train_bytes  # for reporting
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2)
    print(f"data: {len(data):,} bytes | train_chunks={len(train_ds):,} | "
          f"val_chunks={len(val_ds):,} | vocab={vocab}", flush=True)

    results = {}
    if not args.skip_prism:
        prism = build_prism(vocab, args)
        bpc, n = train_model(prism, "PRISM-scaled", train_loader, val_loader,
                             args, device, is_prism=True)
        results["PRISM"] = (bpc, n)
        del prism
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    if not args.skip_mamba:
        md = args.mamba_d or args.d
        mamba = MambaStack(vocab, d_model=md, n_layers=args.n_layers)
        bpc, n = train_model(mamba, f"Mamba-d{md}", train_loader, val_loader,
                             args, device, is_prism=False)
        results["Mamba"] = (bpc, n)

    print("\n" + "=" * 64)
    print("Stage 4 요약 (enwik8 char-LM, val bits-per-char):")
    for k, (bpc, n) in results.items():
        print(f"  {k:14s}: {bpc:.4f} bpc  ({n:,} params)")
    if "PRISM" in results and "Mamba" in results:
        pb, _ = results["PRISM"]; mb, _ = results["Mamba"]
        win = "PRISM" if pb < mb else "Mamba"
        print(f"  → {win} 우세 (Δ {abs(pb-mb):.4f} bpc)")


if __name__ == "__main__":
    main()
