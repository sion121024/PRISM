"""
Stage 3: K-scale validation — 이중시계 핵심 가설 검증.

"내부 반복 K를 늘리면 추론 태스크 성능이 오르는가?"

K=1, K=2, K=4, K=8 on associative recall.
결과가 단조증가하면 이중시계 가설 지지.
"""

import math
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import AssocRecallDataset
from tasks.assoc_recall import AssocRecallModel

N_PAIRS     = 4
KEY_VOCAB   = 16
VAL_VOCAB   = 16
EPOCHS      = 30
BATCH_SIZE  = 128
D           = 128
EMB_DIM     = 32
DEVICE      = "cpu"

K_VALUES = [1, 2, 4, 8]

print("Stage 3: K-scale validation (associative recall)")
print(f"  n_pairs={N_PAIRS}, d={D}, epochs={EPOCHS}, B={BATCH_SIZE}")
print("=" * 60)

results = {}

for K in K_VALUES:
    t0 = time.time()
    dataset = AssocRecallDataset(10000, N_PAIRS, KEY_VOCAB, VAL_VOCAB)
    val_ds  = AssocRecallDataset(1000,  N_PAIRS, KEY_VOCAB, VAL_VOCAB, seed=99)
    loader  = dataset.get_loader(batch_size=BATCH_SIZE)
    val_ldr = val_ds.get_loader(batch_size=BATCH_SIZE, shuffle=False)

    base = PRISMLangModel(
        vocab_size=dataset.total_vocab,
        d=D, emb_dim=EMB_DIM, K=K, alpha=0.05,
        lam=0.01, mem_eta=0.01, mem_gamma=0.001,
        memory_mode="sliding", mem_rank=8,
        use_urec=True, simple_prior=True,
    )
    model = AssocRecallModel(base, val_vocab_offset=KEY_VOCAB)
    params = base.num_params()

    optimizer = optim.AdamW(base.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(loader))

    best_acc = 0.0
    epoch_accs = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for tokens, labels in loader:
            out = model(tokens, labels)
            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        acc = 0.0
        with torch.no_grad():
            for tokens, labels in val_ldr:
                acc += model(tokens, labels)['acc']
        acc /= len(val_ldr)
        epoch_accs.append(acc)
        if acc > best_acc:
            best_acc = acc

    elapsed = time.time() - t0
    results[K] = {'best_acc': best_acc, 'final_acc': epoch_accs[-1],
                  'epoch_accs': epoch_accs, 'elapsed': elapsed, 'params': params}
    print(f"K={K:2d} | best_acc={best_acc:.4f} | final={epoch_accs[-1]:.4f}"
          f" | {elapsed:.1f}s | {params:,} params")

print()
print("=" * 60)
print("K-scale summary:")
for K in K_VALUES:
    r = results[K]
    print(f"  K={K:2d}: best={r['best_acc']:.4f}  ({r['elapsed']:.0f}s)")

# 단조 증가 여부 확인
accs = [results[K]['best_acc'] for K in K_VALUES]
monotone = all(accs[i] <= accs[i+1] for i in range(len(accs)-1))
print(f"\n이중시계 가설 {'✓ 지지 (단조증가)' if monotone else '✗ 불지지 (비단조)'}")
print(f"K=1→8 개선: {(accs[-1] - accs[0])*100:+.1f}%p")
