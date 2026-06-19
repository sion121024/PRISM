"""
PRISM 핵심 사용 예제 (자체 완결, 외부 데이터 불필요).

작은 합성 텍스트로 문자 단위 LM 학습 → 손실 감소 → 생성으로 재현.
표준 API(model(tokens) / model.generate())를 그대로 보여준다.

  python example.py
"""

import torch
import torch.optim as optim

from prism import PRISMLangModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # --- 합성 코퍼스 (문자 단위) ---
    text = ("the quick brown fox jumps over the lazy dog. "
            "prism descends an energy to think. ") * 200
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    data = torch.tensor([stoi[c] for c in text])

    def get_batch(batch, length):
        ix = torch.randint(0, len(data) - length - 1, (batch,))
        return torch.stack([data[i:i + length + 1] for i in ix]).to(device)

    # --- 모델: 에너지 기반 순환 셀 ---
    model = PRISMLangModel(
        vocab_size=vocab, d=128, emb_dim=32,
        K=3,                    # 내부시계: 토큰마다 에너지 K번 하강 (사고 깊이)
        memory_mode="sliding",  # Hebbian 빠른가중치 (저랭크)
        mem_rank=16, simple_prior=True, use_urec=True,
        use_gate=True, use_conv=True,
    ).to(device)
    print(f"vocab={vocab} | PRISM params: {model.num_params():,} | device={device}")

    # --- 학습 루프 (표준 API: out = model(tokens)) ---
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for step in range(1, 401):
        out = model(get_batch(48, 48))      # {'loss', 'logits'}
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:3d} | loss {out['loss'].item():.4f}")

    # --- 생성 (토큰당 O(1), 문맥 길이와 무관) ---
    model.eval()
    seed = "the quick"
    ids = torch.tensor([[stoi[c] for c in seed]], device=device)
    gen = model.generate(ids, max_new_tokens=60, temperature=0.4,
                         top_k=10, repetition_penalty=1.1)
    text_out = "".join(chars[i] for i in gen[0].tolist())
    print(f"\nseed: {seed!r}")
    print(f"gen : {text_out!r}")


if __name__ == "__main__":
    main()
