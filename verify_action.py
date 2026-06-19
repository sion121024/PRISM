"""
행동 슬롯 검증: 텍스트 + 시각 + 행동을 하나의 에너지 E로 통합.

설계철학 검증 — "지각·기억·추론·행동은 같은 E를 하강."

태스크: 멀티모달 결정 (Multimodal Decision)
  - 이미지: 숫자 d ∈ {0..9} 패턴 (시각으로만 알 수 있음)
  - 텍스트: 피연산자 k ∈ {0..4} 를 나타내는 단어 ("zero".."four")
  - 행동:   a = (d + k) mod N_ACT  (시각·텍스트 둘 다 필요)

행동이 두 모달리티의 통합을 요구 → 하나라도 빠지면 풀 수 없음.
같은 상태 x가 같은 에너지를 하강해 텍스트·시각을 동시에 설명하고
그로부터 행동 g_act(x*)를 읽어냄.

Ablation:
  A) Full       : 시각 항 ON, 행동 에너지 결합 ON
  B) No-Vision  : 시각 항 OFF → d 모름 → 행동 ≈ 우연(1/N)
  C) Readout    : 행동을 에너지에 결합하지 않고 순수 readout head로만 학습

실행:  python verify_action.py [--device cuda] [--epochs N] [--scale]
"""

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from prism.agent import PRISMAgentModel

DIGIT_WORDS = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]
N_OPERANDS = 5           # k ∈ {0..4}
N_ACT = 5                # 행동 클래스 수


def build_vocab():
    chars = sorted(set(" ".join(DIGIT_WORDS[:N_OPERANDS])))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["<pad>"] = 0
    return stoi


STOI = build_vocab()
VOCAB_SIZE = len(STOI)


def make_digit_image(digit: int, size: int = 28) -> torch.Tensor:
    img = torch.zeros(1, size, size)
    for i in range(size):
        for j in range(size):
            val = math.sin(i * (digit + 1) * 0.4) * math.cos(j * (digit + 1) * 0.3)
            img[0, i, j] = (val + 1) / 2
    return img


class DecisionDataset(Dataset):
    def __init__(self, n_samples=4000, seq_len=6, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.data = []
        # 이미지 캐시 (디짓별 1개, 결정론적 패턴)
        imgs = [make_digit_image(d) for d in range(10)]
        for _ in range(n_samples):
            d = torch.randint(0, 10, (), generator=g).item()
            k = torch.randint(0, N_OPERANDS, (), generator=g).item()
            word = DIGIT_WORDS[k]
            toks = [STOI[c] for c in word][:seq_len]
            toks += [0] * (seq_len - len(toks))
            tokens = torch.tensor(toks, dtype=torch.long)
            action = (d + k) % N_ACT
            self.data.append((tokens, imgs[d], action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    tokens = torch.stack([b[0] for b in batch])
    images = torch.stack([b[1] for b in batch])
    actions = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return tokens, images, actions


def run(model, loader, vloader, epochs, name, device,
        use_vision=True, couple_action=True):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_acc, best_ppl = 0.0, float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        for tokens, images, actions in loader:
            tokens, images, actions = tokens.to(device), images.to(device), actions.to(device)
            out = model(tokens, images, actions,
                        use_vision=use_vision, couple_action=couple_action)
            loss = out["action_loss"]
            if "text_loss" in out:
                loss = loss + out["text_loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        acc, ppl_sum, nb = 0.0, 0.0, 0
        with torch.no_grad():
            for tokens, images, actions in vloader:
                tokens, images, actions = tokens.to(device), images.to(device), actions.to(device)
                out = model(tokens, images, actions,
                            use_vision=use_vision, couple_action=couple_action)
                acc += out["action_acc"]
                if "text_loss" in out:
                    ppl_sum += math.exp(out["text_loss"].item())
                nb += 1
        acc /= nb; ppl = ppl_sum / nb
        best_acc = max(best_acc, acc); best_ppl = min(best_ppl, ppl)
        print(f"  [{name}] ep {ep:2d} | action_acc {acc:.3f} | text_ppl {ppl:.3f}")
    return best_acc, best_ppl


def make_model(args, device):
    return PRISMAgentModel(
        vocab_size=VOCAB_SIZE, n_actions=N_ACT,
        d=args.d, emb_dim=args.emb_dim, K=args.K,
        mem_rank=args.mem_rank, dec_hidden=args.dec_hidden, vis_dim=args.vis_dim,
        img_size=28, patch_size=7, in_channels=1, use_prior=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_val", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--mem_rank", type=int, default=8)
    p.add_argument("--dec_hidden", type=int, default=64)
    p.add_argument("--vis_dim", type=int, default=32)
    p.add_argument("--scale", action="store_true", help="GPU 스케일업 프리셋")
    args = p.parse_args()

    if args.scale:
        args.d, args.emb_dim, args.K = 256, 64, 6
        args.mem_rank, args.dec_hidden, args.vis_dim = 16, 128, 64
        args.n_train, args.n_val, args.epochs = 20000, 2000, 15
        args.batch_size = 256

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)

    print("=" * 60)
    print("행동 슬롯 검증: 텍스트 + 시각 + 행동 (하나의 에너지 E)")
    print("=" * 60)
    print(f"vocab={VOCAB_SIZE} | 행동=(이미지숫자+텍스트k) mod {N_ACT} | "
          f"우연={1.0/N_ACT:.3f} | device={device}")

    tr = DecisionDataset(args.n_train, seed=0)
    va = DecisionDataset(args.n_val, seed=99)
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    vloader = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"\n--- A: Full (시각 ON, 행동 에너지 결합 ON) ---")
    mA = make_model(args, device)
    print(f"params: {mA.num_params():,}")
    t0 = time.time()
    accA, pplA = run(mA, loader, vloader, args.epochs, "Full", device,
                     use_vision=True, couple_action=True)
    print(f"  ({time.time()-t0:.0f}s)")

    print(f"\n--- B: No-Vision (시각 항 OFF) ---")
    mB = make_model(args, device)
    accB, pplB = run(mB, loader, vloader, args.epochs, "NoVis", device,
                     use_vision=False, couple_action=True)

    print(f"\n--- C: Readout (행동 비결합, 순수 readout head) ---")
    mC = make_model(args, device)
    accC, pplC = run(mC, loader, vloader, args.epochs, "Readout", device,
                     use_vision=True, couple_action=False)

    print("\n" + "=" * 60)
    print("요약 (best action accuracy):")
    print(f"  A Full       : {accA:.3f}   (text_ppl {pplA:.3f})")
    print(f"  B No-Vision  : {accB:.3f}   ← 시각 없으면 d 모름")
    print(f"  C Readout    : {accC:.3f}   ← 행동 에너지 결합 없음")
    print(f"  우연          : {1.0/N_ACT:.3f}")
    print("-" * 60)
    print(f"  시각 항 기여 (A−B): {accA - accB:+.3f}")
    print(f"  행동 결합 기여 (A−C): {accA - accC:+.3f}")
    verdict = "✓ 멀티모달 통합 성공" if (accA - accB > 0.15 and accA > 0.6) \
        else "✗ 통합 약함 (재설계 필요)"
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
