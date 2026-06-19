# PRISM (core) — Predictive Recurrent Implicit State Machine

> "지각·기억·추론·행동은 전부 같은 에너지 함수 E를 경사하강하는 것."
> Perception, memory, reasoning, and action are all gradient descent on one energy E.

PRISM의 **핵심 구조만** 추린 패키지입니다. 고정 크기 연속 상태 `x ∈ ℝ^d`가
에너지 `E(x)`를 내려감으로써 사고를 구현하는 순환 신경망. 외부 의존성은 PyTorch 뿐.

```
prism/
  cell.py        # PRISMCell — 에너지 함수·그래디언트·K-step 내부 반복, Hebbian 기억
  model.py       # PRISMLangModel — 언어 모델 (forward/generate)
  deq.py         # DEQ 솔버 (Anderson + implicit diff, O(1) 메모리 역전파)
  multimodal.py  # PRISMMultimodalModel — 에너지에 시각 항 추가
  agent.py       # PRISMAgentModel — 텍스트+시각+행동을 같은 E의 항으로 통합
example.py       # 자체 완결 예제 (copy 태스크 학습 → 생성)
```

## 설계 철학

```
E(x) = ½‖ũ − g(x)‖²_Π1       ← 지각: 현재 입력 설명
     + ½‖(I−M)x‖²_Π2          ← 기억: Hebbian fast-weight M 과 일치
     + ½‖x − μ(x_prev)‖²_Π3   ← 추론: 이전 상태로부터 예측
     + ½λ‖x‖²                  ← 정규화

dx/ds = −∂E/∂x               (내부시계 s: 토큰마다 K번 = 사고 깊이)
ΔM    = η(ε_mem ⊗ x) − γM     (Hebbian 빠른가중치, backprop 없음)
```

**이중시계**: 외부시계 t(토큰마다 1틱) × 내부시계 s(틱마다 에너지 K번 하강).
쉬운 입력은 K 적게(빠른 반응), 어려운 입력은 K 많이(깊은 사고).

## 설치

```bash
pip install -r requirements.txt   # torch>=2.0
python example.py                 # 동작 확인 (copy 태스크)
```

## 사용법

### 1) 언어 모델

```python
import torch
from prism import PRISMLangModel

model = PRISMLangModel(
    vocab_size=256, d=256, emb_dim=64,
    K=4,                    # 내부 반복(사고 깊이)
    memory_mode="sliding",  # Hebbian 빠른가중치 (저랭크)
    mem_rank=32, simple_prior=True, use_urec=True,
)

tokens = torch.randint(0, 256, (8, 128))      # [batch, seq_len]
out = model(tokens)                            # {'loss', 'logits'}
out["loss"].backward()                         # 느린가중치 θ 학습

# 생성 (RNN처럼 토큰당 O(1) — 문맥 길이와 무관)
gen = model.generate(tokens[:, :10], max_new_tokens=50,
                     temperature=0.7, top_k=40, repetition_penalty=1.2)
```

추론 시 `model.cell.K` 를 바꾸면 같은 가중치로 사고 깊이를 조절할 수 있습니다
(이중시계: K↑ → 에너지 더 깊이 하강).

### 2) 멀티모달 (텍스트 + 시각)

```python
from prism import PRISMMultimodalModel

mm = PRISMMultimodalModel(vocab_size=256, d=128, vis_dim=32,
                          img_size=28, patch_size=7)
out = mm(tokens, images)   # images: [B, T, C, H, W], 없으면 None
```

시각은 별도 fusion layer가 아니라 **에너지에 항 하나가 추가**되는 형태:
`+ ½‖v − g_vis(x)‖²_Π_vis`. K-step 하강이 텍스트·시각을 동시에 만족.

### 3) 에이전트 (텍스트 + 시각 + 행동)

```python
from prism import PRISMAgentModel

agent = PRISMAgentModel(vocab_size=256, n_actions=5, d=128, vis_dim=32)
out = agent(tokens, images=img, action_labels=labels)
# out['action_logits'] — 행동은 g_act(x*)로 상태에서 직접 읽힘 (텍스트 생성 불필요)
```

`prism/agent.py` 의 `ModalSlot` 으로 임의 개수 모달리티를 같은 에너지 항으로 추가 가능.

## 핵심 파라미터

| 파라미터 | 기본 | 설명 |
|---------|------|------|
| `d` | 256 | 상태 차원 |
| `emb_dim` | 64 | 임베딩 차원 |
| `K` | 4 | 내부 반복(사고 깊이) |
| `alpha` | 0.05 | 내부 스텝 크기 |
| `mem_rank` | 32 | 슬라이딩 Hebbian 메모리 rank |
| `simple_prior` | True | identity prior μ=x_prev (0 params) |
| `use_urec` | True | 관측 증강 ũ=f([u, x_prev]) (권장) |
| `approximate_grad` | False | K-1 no_grad + 1 grad (역전파 그래프 1/K) |
| `n_layers` | 1 | 계층적 예측 코딩 레이어 수 |

## 강점과 한계 (정직하게)

**강점**
- 적은 파라미터로 추론·기억이 강함 (명시적 Hebbian 메모리가 부하에 강건).
- 추론 시 **토큰당 O(1) 비용·메모리** — 문맥 길이와 무관 (RNN류).
- 적응형 사고 깊이(이중시계 K), 지각·기억·추론·행동을 한 에너지로 통합.

**한계**
- 학습이 **GPU 비친화적**: 토큰을 순차 처리(재귀)하므로 시퀀스 병렬화가 안 됨
  → 대규모 학습은 Transformer보다 느림. 비선형 에너지 하강이라 Mamba식 parallel
  scan으로도 못 바꿈.
- 따라서 적합한 자리는 *대규모 raw LM*이 아니라 *적은 파라미터·긴 문맥·지각-행동
  통합*이 필요한 영역(엣지/스트리밍/제어).

## 학습 전략

- 느린가중치 θ (D, Π, embed, unembed): K-step unroll 역전파 (또는 DEQ implicit diff로 O(1) 메모리).
- 빠른가중치 M: Hebbian 갱신 `ΔM = η(ε_mem·xᵀ) − γM`, backprop 없음(detach).
