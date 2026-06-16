"""
PRISM 평가 스크립트.

에너지 수렴 시각화 + 태스크 정확도 + 텍스트 생성.

사용법:
  python eval.py --task copy --checkpoint best_prism.pt
  python eval.py --task char_lm --generate "To be"
  python eval.py --task energy_plot
"""

import argparse
import math
import torch
import torch.nn.functional as F

from prism import PRISMLangModel
from tasks import CopyTaskDataset, AssocRecallDataset, TinyShakespeare
from tasks.copy_task import copy_task_accuracy
from tasks.assoc_recall import AssocRecallModel


def eval_energy(args):
    """에너지 궤적 시각화 — 수렴 확인."""
    import matplotlib.pyplot as plt

    device = torch.device(args.device)
    vocab_size = 64
    model = PRISMLangModel(
        vocab_size=vocab_size, d=args.d, emb_dim=args.emb_dim,
        K=args.K, alpha=args.alpha,
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    B = 4
    tokens = torch.randint(0, vocab_size, (B,), device=device)
    u = model.embed(tokens)
    M = torch.zeros(B, args.d, args.d, device=device)

    _, energies = model.cell.iterate(u, M, return_energies=True, K=args.K)

    print(f"E[0]  = {energies[0]:.6f}")
    print(f"E[K]  = {energies[-1]:.6f}")
    print(f"감소량 = {energies[0] - energies[-1]:.6f} ({(1-energies[-1]/energies[0])*100:.1f}%)")

    plt.figure(figsize=(8, 4))
    plt.plot(energies, 'b-o', markersize=4)
    plt.xlabel("Inner step s")
    plt.ylabel("E(x)")
    plt.title("Energy descent during PRISM inner iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eval_energy.png", dpi=150)
    print("저장: eval_energy.png")


def eval_copy(args):
    device = torch.device(args.device)
    ds = CopyTaskDataset(n_samples=1000, seq_len=args.seq_len, vocab_size=32, seed=999)
    loader = ds.get_loader(batch_size=64, shuffle=False)

    model = PRISMLangModel(
        vocab_size=ds.total_vocab, d=args.d, emb_dim=args.emb_dim,
        K=args.K, alpha=args.alpha, lam=args.lam,
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    total_acc = 0.0
    with torch.no_grad():
        for tokens, mask in loader:
            tokens, mask = tokens.to(device), mask.to(device)
            out = model(tokens)
            total_acc += copy_task_accuracy(out['logits'], tokens, mask)
    print(f"Copy task accuracy: {total_acc / len(loader):.4f}")


def eval_char_lm(args):
    device = torch.device(args.device)
    val_ds = TinyShakespeare(block_size=args.block_size, split="val")
    loader = val_ds.get_loader(batch_size=32, shuffle=False)

    model = PRISMLangModel(
        vocab_size=val_ds.vocab_size, d=args.d, emb_dim=args.emb_dim,
        K=args.K, alpha=args.alpha, lam=args.lam,
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for tokens in loader:
            tokens = tokens.to(device)
            out = model(tokens)
            total_loss += out['loss'].item()

    val_ppl = math.exp(total_loss / len(loader))
    print(f"Val perplexity: {val_ppl:.2f}")

    if args.generate:
        prompt_ids = val_ds.encode(args.generate).unsqueeze(0).to(device)
        generated = model.generate(
            prompt_ids, max_new_tokens=200,
            temperature=args.temperature, top_k=args.top_k,
        )
        text = val_ds.decode(generated[0])
        print(f"\n--- 생성 텍스트 ---\n{text}\n---")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["energy", "copy", "assoc_recall", "char_lm"],
                   default="energy")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--generate", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.task == "energy":
        eval_energy(args)
    elif args.task == "copy":
        eval_copy(args)
    elif args.task == "char_lm":
        eval_char_lm(args)
