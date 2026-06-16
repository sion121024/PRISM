"""
Stage-1 sanity check: verify energy converges under inner-clock descent
and that DEQ implicit diff produces non-NaN gradients.

Run BEFORE training to confirm the architecture is alive.
"""

import torch
import torch.nn.functional as F
from prism import PRISMCell, PRISMLM, CharDataset


def check_energy_convergence(d=64, K=32, step=0.05, B=4):
    print("=" * 55)
    print("1. Energy convergence (inner clock)")
    print("=" * 55)
    cell = PRISMCell(d)
    x = torch.randn(B, d) * 0.1
    u = torch.randn(B, d)

    x_out, energies = cell.descend(x, u, K=K, step=step)
    print(f"   E[0]  = {energies[0]:.4f}")
    print(f"   E[-1] = {energies[-1]:.4f}")
    delta = energies[0] - energies[-1]
    ok = delta > 0
    print(f"   ΔE    = {delta:.4f}  {'✓ converging' if ok else '✗ NOT converging'}")
    return ok


def check_deq_grads(vocab=50, d=64, K=8, B=2, T=16):
    print()
    print("=" * 55)
    print("2. DEQ implicit diff — gradient flow")
    print("=" * 55)
    text = "hello world " * 50
    ds   = CharDataset(text, seq_len=T)
    model = PRISMLM(ds.vocab_size, d=d, K=K, step=0.05)

    x = torch.randint(0, ds.vocab_size, (B, T))
    y = torch.randint(0, ds.vocab_size, (B, T))

    logits = model(x, use_deq=True)
    loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    nan_params = [n for n, p in model.named_parameters()
                  if p.grad is not None and p.grad.isnan().any()]
    ok = len(nan_params) == 0
    print(f"   Loss = {loss.item():.4f}")
    print(f"   NaN grads: {nan_params if nan_params else 'none'}  "
          f"{'✓' if ok else '✗'}")
    return ok


def check_memory_update(d=64):
    print()
    print("=" * 55)
    print("3. Fast-weight (M) update — associative recall")
    print("=" * 55)
    cell = PRISMCell(d)
    B = 1
    # Memorise a pattern
    key = torch.randn(B, d)
    u   = torch.randn(B, d)
    x, _ = cell.descend(key, u, K=16, step=0.05)
    cell.update_M(x, eta=0.1, gamma=0.0)

    # Query with similar x: check M·x ≈ x (fixed-point property)
    noise = key + 0.05 * torch.randn_like(key)
    err_before = ((noise - noise @ cell.M.T)**2).mean().item()
    x2, _ = cell.descend(noise, u, K=16, step=0.05)
    err_after  = ((x2 - x2 @ cell.M.T)**2).mean().item()
    ok = err_after < err_before
    print(f"   memory err before descent: {err_before:.4f}")
    print(f"   memory err after  descent: {err_after:.4f}")
    print(f"   {'✓ memory helps' if ok else '✗ memory not helping yet'}")
    return ok


if __name__ == "__main__":
    r1 = check_energy_convergence()
    r2 = check_deq_grads()
    r3 = check_memory_update()
    print()
    all_ok = r1 and r2 and r3
    print("=" * 55)
    print(f"Overall: {'ALL PASS ✓' if all_ok else 'SOME FAILED ✗'}")
    print("=" * 55)
