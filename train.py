"""
PRISM Stage-1 training: character LM on a small text corpus.

Usage:
    python train.py                          # default (tiny Shakespeare)
    python train.py --text path/to/file.txt
    python train.py --K 16                   # deeper thinking per token
    python train.py --K_eval 32 --compare    # compare K=8 vs K=32 at eval
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from prism import PRISMLM, CharDataset


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def get_text(path: str | None) -> str:
    if path:
        return Path(path).read_text()
    cache = Path("data/tinyshakespeare.txt")
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        print("Downloading tiny Shakespeare …")
        urllib.request.urlretrieve(SHAKESPEARE_URL, cache)
    return cache.read_text()


@torch.no_grad()
def evaluate(model, loader, device, K: int | None = None) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, K=K, use_deq=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * x.numel()
        n += x.numel()
    return math.exp(total_loss / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",   default=None)
    parser.add_argument("--d",      type=int,   default=256)
    parser.add_argument("--K",      type=int,   default=8,    help="inner steps (train)")
    parser.add_argument("--K_eval", type=int,   default=None, help="inner steps (eval, default=K)")
    parser.add_argument("--step",   type=float, default=0.05)
    parser.add_argument("--seq",    type=int,   default=128)
    parser.add_argument("--batch",  type=int,   default=32)
    parser.add_argument("--epochs", type=int,   default=20)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--deq",    action="store_true", default=True)
    parser.add_argument("--no_deq", dest="deq", action="store_false")
    parser.add_argument("--compare", action="store_true",
                        help="compare ppl at K vs K_eval at each eval")
    parser.add_argument("--save",   default="checkpoints/prism.pt")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    text   = get_text(args.text)
    ds     = CharDataset(text, seq_len=args.seq)
    n_val  = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)

    model = PRISMLM(
        vocab_size=ds.vocab_size,
        d=args.d,
        K=args.K,
        step=args.step,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab: {ds.vocab_size}  |  Params: {n_params:,}  |  K={args.K}  |  DEQ={args.deq}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    K_eval = args.K_eval or args.K
    Path(args.save).parent.mkdir(exist_ok=True)
    best_ppl = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, K=args.K, use_deq=args.deq)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_steps += 1

        train_loss = total_loss / n_steps
        val_ppl    = evaluate(model, val_loader, device, K=K_eval)
        elapsed    = time.time() - t0

        line = (f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_ppl(K={K_eval})={val_ppl:.2f}  "
                f"({elapsed:.1f}s)")

        if args.compare and args.K_eval and args.K_eval != args.K:
            ppl_base = evaluate(model, val_loader, device, K=args.K)
            line += f"  val_ppl(K={args.K})={ppl_base:.2f}"

        print(line)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "vocab": {"stoi": ds.stoi, "itos": ds.itos}},
                       args.save)

    print(f"\nBest val ppl: {best_ppl:.2f}  |  saved to {args.save}")


if __name__ == "__main__":
    main()
