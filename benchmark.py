"""
Benchmark: PRISM vs GRU baseline on char LM and associative recall.

Usage:
    python benchmark.py --text data/tinyshakespeare.txt
    python benchmark.py --quick   # small subset, fast
"""

import argparse
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from prism import PRISMLM, CharDataset
from baselines import GRULM
from tasks import AssociativeRecallDataset


def train_epoch(model, loader, opt, device, K=None):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, PRISMLM):
            logits = model(x, K=K, use_deq=False)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item(); n += 1
    return total / n


@torch.no_grad()
def eval_ppl(model, loader, device, K=None):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, PRISMLM):
            logits = model(x, K=K, use_deq=False)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item() * x.numel(); n += x.numel()
    return math.exp(total / n)


@torch.no_grad()
def eval_recall(model, loader, device, seq_len, K=None):
    """Accuracy on associative recall: only the last token prediction counts."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, PRISMLM):
            logits = model(x, K=K, use_deq=False)
        else:
            logits = model(x)
        pred = logits[:, -1].argmax(-1)
        correct += (pred == y).sum().item()
        total   += y.shape[0]
    return correct / total


def run_lm_benchmark(args, device):
    print("\n" + "=" * 60)
    print("CHAR LM BENCHMARK  (tiny Shakespeare)")
    print("=" * 60)

    text = open(args.text).read()
    if args.quick:
        text = text[:50_000]
    ds = CharDataset(text, seq_len=args.seq)
    n_val = max(1, len(ds) // 10)
    tr, va = random_split(ds, [len(ds)-n_val, n_val],
                          generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,  drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False)

    models = {
        f"PRISM(K={args.K})": PRISMLM(ds.vocab_size, d=args.d, K=args.K).to(device),
        "GRU(2L)":            GRULM(ds.vocab_size,   d=args.d, n_layers=2).to(device),
    }

    for name, model in models.items():
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        print(f"\n{name}  params={sum(p.numel() for p in model.parameters()):,}")
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            train_epoch(model, tl, opt, device,
                        K=args.K if isinstance(model, PRISMLM) else None)
        ppl = eval_ppl(model, vl, device,
                       K=args.K if isinstance(model, PRISMLM) else None)
        elapsed = time.time() - t0
        print(f"  val_ppl={ppl:.2f}  time={elapsed:.1f}s")

        if isinstance(model, PRISMLM):
            ppl_deep = eval_ppl(model, vl, device, K=args.K * 4)
            print(f"  val_ppl(K={args.K*4}, no retrain)={ppl_deep:.2f}  "
                  f"{'↓ deeper=better ✓' if ppl_deep < ppl else '↑ deeper≠better'}")


def run_recall_benchmark(args, device):
    print("\n" + "=" * 60)
    print("ASSOCIATIVE RECALL BENCHMARK")
    print("=" * 60)

    ds  = AssociativeRecallDataset(vocab_size=16, n_pairs=4, n_samples=4000)
    n_val = 400
    tr, va = random_split(ds, [len(ds)-n_val, n_val],
                          generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=64, shuffle=True,  drop_last=True)
    vl = DataLoader(va, batch_size=64, shuffle=False)

    models = {
        f"PRISM(K={args.K})": PRISMLM(ds.vocab_size, d=args.d, K=args.K).to(device),
        "GRU(2L)":            GRULM(ds.vocab_size,   d=args.d, n_layers=2).to(device),
    }

    for name, model in models.items():
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        print(f"\n{name}")
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                if isinstance(model, PRISMLM):
                    logits = model(x, K=args.K, use_deq=False)
                    loss   = F.cross_entropy(logits[:, -1], y)
                else:
                    logits = model(x)
                    loss   = F.cross_entropy(logits[:, -1], y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        acc = eval_recall(model, vl, device, ds.seq_len,
                          K=args.K if isinstance(model, PRISMLM) else None)
        print(f"  recall_acc={acc*100:.1f}%")

        if isinstance(model, PRISMLM):
            acc_deep = eval_recall(model, vl, device, ds.seq_len, K=args.K * 4)
            print(f"  recall_acc(K={args.K*4})={acc_deep*100:.1f}%  "
                  f"{'↑ deeper=better ✓' if acc_deep > acc else '↓ deeper≠better'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",   default="data/tinyshakespeare.txt")
    parser.add_argument("--d",      type=int,   default=128)
    parser.add_argument("--K",      type=int,   default=8)
    parser.add_argument("--epochs", type=int,   default=10)
    parser.add_argument("--batch",  type=int,   default=64)
    parser.add_argument("--seq",    type=int,   default=64)
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    run_recall_benchmark(args, device)
    run_lm_benchmark(args, device)


if __name__ == "__main__":
    main()
