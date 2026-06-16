"""
PRISM Stage 1 학습 스크립트.

사용법:
  python train.py --task copy
  python train.py --task assoc_recall
  python train.py --task char_lm
  python train.py --task char_lm --baseline lstm
"""

import argparse
import time
import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# CPU 소규모 텐서에서 스레드 동기화 오버헤드 제거 (136x speedup)
torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import CopyTaskDataset, AssocRecallDataset, TinyShakespeare
from tasks.assoc_recall import AssocRecallModel
from tasks.copy_task import copy_task_accuracy
from baselines import LSTMLangModel


# ------------------------------------------------------------------ #
# 학습 루프                                                            #
# ------------------------------------------------------------------ #

def train_copy(args):
    device = torch.device(args.device)

    dataset = CopyTaskDataset(
        n_samples=10000, seq_len=args.seq_len, vocab_size=32,
    )
    val_dataset = CopyTaskDataset(
        n_samples=1000, seq_len=args.seq_len, vocab_size=32, seed=99,
    )
    loader = dataset.get_loader(batch_size=args.batch_size)
    val_loader = val_dataset.get_loader(batch_size=args.batch_size, shuffle=False)

    model = PRISMLangModel(
        vocab_size=dataset.total_vocab,
        d=args.d, emb_dim=args.emb_dim,
        K=args.K, alpha=args.alpha,
        lam=args.lam, mem_eta=args.mem_eta, mem_gamma=args.mem_gamma,
        memory_mode=args.memory_mode, mem_rank=args.mem_rank,
        approximate_grad=args.approximate_grad,
    ).to(device)

    print(f"PRISM params: {model.num_params():,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for tokens, mask in loader:
            tokens, mask = tokens.to(device), mask.to(device)
            out = model(tokens)
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # 검증
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for tokens, mask in val_loader:
                tokens, mask = tokens.to(device), mask.to(device)
                out = model(tokens)
                val_acc += copy_task_accuracy(out['logits'], tokens, mask)
        val_acc /= len(val_loader)

        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.3f}"
        )


def train_assoc_recall(args):
    device = torch.device(args.device)
    n_pairs = args.n_pairs
    key_vocab = 16
    val_vocab = 16

    dataset = AssocRecallDataset(10000, n_pairs, key_vocab, val_vocab)
    val_dataset = AssocRecallDataset(1000, n_pairs, key_vocab, val_vocab, seed=99)
    loader = dataset.get_loader(batch_size=args.batch_size)
    val_loader = val_dataset.get_loader(batch_size=args.batch_size, shuffle=False)

    base_model = PRISMLangModel(
        vocab_size=dataset.total_vocab,
        d=args.d, emb_dim=args.emb_dim,
        K=args.K, alpha=args.alpha,
        lam=args.lam, mem_eta=args.mem_eta, mem_gamma=args.mem_gamma,
        memory_mode=args.memory_mode, mem_rank=args.mem_rank,
        approximate_grad=args.approximate_grad,
    ).to(device)
    model = AssocRecallModel(base_model, val_vocab_offset=key_vocab).to(device)

    print(f"PRISM params: {base_model.num_params():,}")

    optimizer = optim.AdamW(
        base_model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            out = model(tokens, labels)
            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out['loss'].item()

        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                out = model(tokens, labels)
                val_acc += out['acc']
        val_acc /= len(val_loader)

        print(
            f"Epoch {epoch:3d} | loss {total_loss/len(loader):.4f}"
            f" | val_acc {val_acc:.3f}"
        )


def train_char_lm(args):
    device = torch.device(args.device)
    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds = TinyShakespeare(block_size=args.block_size, split="val")
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)

    vocab_size = train_ds.vocab_size

    if args.baseline == "lstm":
        model = LSTMLangModel(
            vocab_size=vocab_size, emb_dim=args.emb_dim,
            hidden_dim=args.d, n_layers=2,
        ).to(device)
        print(f"LSTM params: {model.num_params():,}")
    else:
        model = PRISMLangModel(
            vocab_size=vocab_size, d=args.d, emb_dim=args.emb_dim,
            K=args.K, alpha=args.alpha, lam=args.lam,
            mem_eta=args.mem_eta, mem_gamma=args.mem_gamma,
            memory_mode=args.memory_mode, mem_rank=args.mem_rank,
            approximate_grad=args.approximate_grad,
        ).to(device)
        print(f"PRISM params: {model.num_params():,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_val_ppl = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for tokens in train_loader:
            tokens = tokens.to(device)
            out = model(tokens, tbptt_window=args.tbptt)
            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out['loss'].item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tokens in val_loader:
                tokens = tokens.to(device)
                out = model(tokens)
                val_loss += out['loss'].item()
        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), f"best_{args.baseline or 'prism'}.pt")

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss {total_loss/len(train_loader):.4f} | "
            f"val_ppl {val_ppl:.2f} | "
            f"{elapsed:.1f}s"
        )

    print(f"\nBest val ppl: {best_val_ppl:.2f}")


# ------------------------------------------------------------------ #
# argparse                                                            #
# ------------------------------------------------------------------ #

def get_args():
    p = argparse.ArgumentParser(description="PRISM Stage 1 Training")

    p.add_argument("--task", choices=["copy", "assoc_recall", "char_lm"],
                   default="char_lm")
    p.add_argument("--baseline", choices=["lstm"], default=None,
                   help="char_lm 에서만 사용: lstm 베이스라인")

    # 모델
    p.add_argument("--d", type=int, default=256, help="상태 차원")
    p.add_argument("--emb_dim", type=int, default=64, help="임베딩 차원")
    p.add_argument("--K", type=int, default=4, help="내부 반복 횟수")
    p.add_argument("--alpha", type=float, default=0.05, help="내부 스텝 크기")
    p.add_argument("--memory_mode", choices=["sliding","full_M","none"],
                   default="sliding")
    p.add_argument("--mem_rank", type=int, default=8, help="슬라이딩 메모리 rank")
    p.add_argument("--approximate_grad", action="store_true", default=False,
                   help="K-1 no_grad + 1 grad 근사 (CPU 최적화)")
    p.add_argument("--lam", type=float, default=0.01, help="정규화 λ")
    p.add_argument("--mem_eta", type=float, default=0.01, help="M 학습률 η")
    p.add_argument("--mem_gamma", type=float, default=0.001, help="M 감쇠 γ")

    # 학습
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tbptt", type=int, default=0, help="TBPTT 윈도우 (0=전체)")

    # 태스크별
    p.add_argument("--seq_len", type=int, default=10, help="copy task 길이")
    p.add_argument("--n_pairs", type=int, default=4, help="assoc_recall 쌍 수")
    p.add_argument("--block_size", type=int, default=128, help="char_lm 블록 크기")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"Device: {args.device} | Task: {args.task}")

    if args.task == "copy":
        train_copy(args)
    elif args.task == "assoc_recall":
        train_assoc_recall(args)
    elif args.task == "char_lm":
        train_char_lm(args)
