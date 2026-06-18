"""
다층 PRISM 검증: n_layers=1 vs 2 on char_lm.

계층적 예측 코딩이 단일 레이어보다 낮은 perplexity를 달성하는가?
같은 파라미터 예산에서 비교.
"""

import math
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare

EPOCHS     = 5
BATCH_SIZE = 32
BLOCK_SIZE = 128
D          = 256
EMB_DIM    = 64
K          = 4
DEVICE     = "cpu"

train_ds = TinyShakespeare(block_size=BLOCK_SIZE, split="train")
val_ds   = TinyShakespeare(block_size=BLOCK_SIZE, split="val")
train_ldr = train_ds.get_loader(batch_size=BATCH_SIZE)
val_ldr   = val_ds.get_loader(batch_size=BATCH_SIZE, shuffle=False)
V = train_ds.vocab_size

def run(name, model):
    print(f"\n--- {name} | {model.num_params():,} params ---")
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_ldr))
    best_ppl = float('inf')
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tl = 0.0
        for tokens in train_ldr:
            out = model(tokens)
            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tl += out['loss'].item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for tokens in val_ldr:
                vl += model(tokens)['loss'].item()
        vl /= len(val_ldr)
        ppl = math.exp(vl)
        if ppl < best_ppl:
            best_ppl = ppl
        print(f"  Epoch {epoch} | train_loss={tl/len(train_ldr):.4f}"
              f" | val_ppl={ppl:.2f} | {time.time()-t0:.1f}s")
    print(f"  Best ppl: {best_ppl:.2f}")
    return best_ppl

configs = [
    ("1-layer (baseline)",
     PRISMLangModel(vocab_size=V, d=D, emb_dim=EMB_DIM, K=K,
                    memory_mode="sliding", mem_rank=8,
                    use_urec=True, simple_prior=True)),
    ("2-layer (hierarchical)",
     PRISMLangModel(vocab_size=V, d=D, emb_dim=EMB_DIM, K=K,
                    memory_mode="sliding", mem_rank=8,
                    use_urec=True, simple_prior=True, n_layers=2)),
    ("1-layer + gate",
     PRISMLangModel(vocab_size=V, d=D, emb_dim=EMB_DIM, K=K,
                    memory_mode="sliding", mem_rank=8,
                    use_urec=True, simple_prior=True, use_gate=True)),
]

print("다층 PRISM 검증 (char_lm, TinyShakespeare)")
print(f"  d={D}, emb_dim={EMB_DIM}, K={K}, epochs={EPOCHS}")
print("=" * 60)

results = {}
for name, model in configs:
    results[name] = run(name, model)

print()
print("=" * 60)
print("요약:")
for name, ppl in results.items():
    print(f"  {name}: {ppl:.2f} ppl")
