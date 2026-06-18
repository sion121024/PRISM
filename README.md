# PRISM — Predictive Recurrent Implicit State Machine

> "지각·기억·추론·행동은 전부 같은 에너지 함수 E를 경사하강하는 것."

고정 크기 연속 상태 x ∈ ℝ^d가 에너지 E(x)를 내려감으로써 사고를 구현하는 순환 신경망 아키텍처.

---

## 설계 철학

```
E(x) = ½‖ũ − g(x)‖²_Π1       ← 지각: 현재 입력 설명
     + ½‖(I−M)x‖²_Π2          ← 기억: Hebbian fast-weight와 일치
     + ½‖x − μ(x_prev)‖²_Π3   ← 추론: 이전 상태로부터 예측
     + ½λ‖x‖²                  ← 정규화

dx/ds = −∂E/∂x   (K번 반복 = 내부 사고 깊이)
ΔM    = η(εmem ⊗ x) − γM      (비대칭 Hebbian: key=x̂, value=ê_mem)
ũ     = f([u_raw, x_prev])     (관측 증강: 현재 입력 + 이전 상태 융합)
```

### 이중 시계 (Dual Clock)

| 시계 | 틱 | 역할 |
|------|-----|------|
| 외부 t | 토큰마다 1회 | 입력 처리, Hebbian 갱신 |
| 내부 s | 토큰마다 K회 | 에너지 하강 = 사고 |

**K가 클수록 더 깊은 추론** — 어려운 토큰에 더 많은 K를 배분하는 적응형 K(t) 지원.

### 핵심 원칙

| 원칙 | 구현 |
|------|------|
| 모든 계산 = 에너지 하강 | carry gate 제거, prior 항으로 대체 |
| 비대칭 Hebbian 기억 | key=normalize(x), value=normalize(εmem) |
| Prior 항 필수 | μ(x_prev)가 없으면 K-effect 없음 |
| 비볼록 에너지 | MLP decoder g(x) → K-step이 실제 추론 수행 |

---

## 실험 결과

### TinyShakespeare 문자 LM (char-level perplexity)

**데이터**: 셰익스피어 전집 ~1.1MB, vocab 65자, block_size=64

#### Stage 12: Mamba 격차 원인 분석 (3 epochs, 조기 신호)

| 모델 | K | rank | 3 epoch BEST ppl | 속도 | 비고 |
|------|---|------|-----------------|------|------|
| PRISM-slim-K4 | 4 | 24 | 28.381 | 261s/epoch | 기준 |
| PRISM-slim-K8 | 8 | 24 | 28.251 | 507s/epoch | +0.129 ppl |
| PRISM-highrank-K4 | 4 | 48 | 28.373 | 375s/epoch | +0.007 ppl |
| **Mamba** | — | — | **7.850** | 75s/epoch | −20.5 ppl 격차 |

- **K=8 효과**: +0.129 ppl 개선, 학습 2× 느림 (3 epoch 기준; 장기에서 더 큰 효과 예상)
- **rank=48 효과**: +0.007 ppl (무의미) — 초기 학습에서 기억 미충전으로 rank 증가 효과 없음
- **Mamba 조기 우위**: epoch 1부터 **9.381 ppl** — PRISM epoch 1 (28.466)의 3배 낮음
- **핵심 발견**: Mamba conv1d가 즉시 로컬 n-gram 패턴 포착; PRISM은 Hebbian 기억 충전에 수 epoch 필요
- **동기**: `use_conv=True` 추가 → PRISM도 epoch 1부터 로컬 패턴 학습 가능 (단 320 params)

#### Stage 14: 누적 개선사항 ablation (5 epochs, norm_f 포함)

| 모델 | params | 5 epoch BEST ppl | 비고 |
|------|--------|-----------------|------|
| slim-K4 (기준) | 55,594 | 28.156 | 5 epoch 내내 flat |
| slim+sel-K4 | 70,674 | **26.070** | input_dep_pi: 후반부 급격 개선 |
| slim+mom-K4 | 55,594 | 28.103 | momentum 단독: 효과 미미 |
| slim+all-K4 | 70,842 | (진행 중) | sel+mom+prior_bias 조합 |
| slim+conv-K4 | 55,914 | (진행 중) | **핵심 테스트**: conv 단독 효과 |
| slim+conv+all-K4 | 71,162 | (진행 중) | conv+sel+mom+prior_bias 전체 |
| Mamba (참조) | 54,400 | (진행 중) | ~7.85 예상 |

#### Stage 10: 단순화 실험 (simple_prior vs prior_mu MLP)

| 모델 | prior | params | BEST ppl |
|------|-------|--------|----------|
| PRISM-v1-K4 | 학습 MLP | 55,452 | **12.794** |
| PRISM-slim-K2 | identity (μ=x_prev) | 55,258 | 17.359 |
| PRISM-slim-K4 | identity (μ=x_prev) | 55,258 | 15.048 |

- **단순화 비용**: v1-K4 vs slim-K4 → **+2.254 ppl** (10 epoch 기준)
- **K-effect (slim)**: slim K2→K4 → **+2.311 ppl** (학습 prior의 0.66ppl 대비 3.5×)
- **핵심 발견**: identity prior가 에너지 경관을 더 어렵게 만들어 K-effect가 3.5× 커짐
- **수렴 패턴**: slim은 epoch 1-4 정체(~28 ppl) → epoch 4-5 돌파 → 지속 개선 (epoch 10에도 수렴 중)
- **u_rec 필수 확인**: u_rec 제거 시 slim이 28 ppl 정체 (학습 실패) — u_rec은 제거 불가

#### Stage 8: 파라미터 매칭 최종 비교

동일 파라미터 예산(~55K)에서 공정 비교:

| 모델 | params | 5 epoch ppl | 8 epoch ppl |
|------|--------|------------|------------|
| PRISM-prior-K2 | 55,452 | 13.601 | ✓ |
| **PRISM-prior-K4** | **55,452** | **12.942** | ✓ |
| LSTM | 56,080 | 14.164 | 기준 |

- **K-effect**: K2→K4 **+0.659 ppl** (동일 파라미터, "더 많이 생각 = 더 똑똑" 증명)
- **PRISM vs LSTM**: K4 기준 **−1.222 ppl 우위** (파라미터 628 차이, ~1% 이내)

#### Stage 7: K-effect 증명 (116K params)

| 구성 | val ppl | K-effect |
|------|---------|----------|
| O-prior-K2 | 16.715 | — |
| P-prior-K4 | **14.951** | +1.765 ppl |
| LSTM (참조) | 20.542 | — |

**K2→K4: 1.765 ppl 개선** — "더 많이 생각 = 더 똑똑" 실증.

#### Ablation 요약

| 조건 | val ppl | 핵심 발견 |
|------|---------|----------|
| linear decoder | ~30 | E 볼록 → K 무의미 |
| MLP decoder, prior 없음 | ~28 | K-effect 없음 (K2≈K4≈K8) |
| MLP decoder + carry gate (설계 비정합) | 22.9 | K-effect 발생, but carry gate = 에너지 밖 변환 |
| MLP decoder + prior (설계 정합) | **14.9** | K-effect 1.7+ ppl, 설계 원칙 준수 |

> **Prior 항이 K-effect의 핵심**: prior 없이는 K=2,4,8 모두 ~28 ppl로 동일.
> **비대칭 Hebbian**: 대칭 Hopfield→비대칭으로 바꿨을 때 K-effect가 0.04→1.74로 43배 증가.

---

## 아키텍처

```
prism/
  cell.py           # PRISMCell — 에너지 함수, 그래디언트, K-step
  deq.py            # DEQ 솔버 (Anderson acceleration)
  model.py          # PRISMLangModel — 전체 언어 모델
  multimodal.py     # PRISMMultimodalModel — 멀티모달 확장
tasks/
  char_lm.py        # TinyShakespeare (문자 LM)
  copy_task.py
  assoc_recall.py
baselines/
  lstm_lm.py
  mamba_lm.py       # Mamba (S6, 순수 PyTorch)
stage3_ablation.py       # 비대칭 Hebbian + carry gate ablation
stage4_design.py         # 설계 정합 검증
stage7_prior.py          # Prior K-effect 실증
stage8_param_match.py    # 파라미터 매칭 최종 비교 (55K)
stage9_mamba_compare.py  # Mamba 첫 비교
stage10_slim.py          # 단순화: simple_prior
stage10c_prior_bias.py   # prior_bias 효과 검증
stage11_adaptive_k.py    # 적응형 K(t) 실측
stage12_gap_analysis.py  # Mamba 격차: K↑, rank↑ 효과
stage13_selective_pi.py  # Selective PRISM: Π(u) vs const Π
stage14_fast_compare.py  # 전체 개선사항 ablation (5 epoch 신호)
verify_adaptive_k.py
verify_convergence.py
verify_multimodal.py
```

---

## 실행

```bash
# 파라미터 매칭 최종 비교 (PRISM 55K vs LSTM 56K)
python stage8_param_match.py --epochs 10 --block_size 64

# K-effect 증명 (prior 항, 116K)
python stage7_prior.py --epochs 5 --block_size 64

# 적응형 K 검증 (어려운 토큰 = 더 많은 K-step)
python verify_adaptive_k.py --train_epochs 3 --K_max 8

# 에너지 수렴 확인
python verify_convergence.py
```

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `d` | 256 | 상태 차원 |
| `emb_dim` | 64 | 임베딩 차원 |
| `K` | 4 | 내부 반복 횟수 (사고 깊이) |
| `alpha` | 0.05 | 내부 스텝 크기 |
| `mem_scale` | 4.0 | Hebbian 메모리 강도 |
| `mem_rank` | 32 | 슬라이딩 메모리 rank |
| `use_prior` | False | Prior 항 (학습 MLP μ(x_prev), +18K params) |
| `simple_prior` | True | Identity prior (μ = x_prev, 0 params) |
| `prior_bias` | False | Biased prior: μ = x_prev + b (+d params, 빠른 수렴) |
| `use_urec` | True | 관측 증강 ũ = f([u_raw, x_prev]) (필수) |
| `input_dep_pi` | False | 선택적 precision Π1(u), Π2(u) — Mamba 유사체 |
| `momentum` | 0.0 | K-step Heavy-ball β (0.9 권장, 0=끔) |
| `use_conv` | False | Depthwise conv1d n-gram 패턴 캡처 (+320 params, Mamba 유사체) |
| `d_conv` | 4 | conv1d 커널 크기 |
| `use_gate` | False | Z-gate: x = x × SiLU(W_z·u) — Mamba y×SiLU(z) 유사체 (+10,920 params) |
| `use_bypass` | False | n-gram 단축로: logits += W_bypass·u_conv (+4,160 params) |

### Selective PRISM: input_dep_pi

PRISM의 고정 precision (Π1, Π2) → 입력 의존 precision Π1(u), Π2(u).

| 설계 | Mamba | PRISM (Selective) |
|------|-------|-------------------|
| 선택적 입력 통합 | B(x_t) ∈ ℝ^(d×N) | Π1(u) ∈ ℝ^emb — 지각 가중치 |
| 선택적 메모리 읽기 | C(x_t) ∈ ℝ^(N×d) | Π2(u) ∈ ℝ^d — 기억 precision |
| 추가 파라미터 | — | emb² + emb×d ≈ +15K |

```python
# 기준: 고정 precision (모든 토큰 동일 가중치)
m = PRISMLangModel(..., input_dep_pi=False)

# Selective PRISM: Mamba 유사 선택적 처리
m = PRISMLangModel(..., input_dep_pi=True)

# K-step Heavy-ball 추가 (파라미터 0 추가)
m = PRISMLangModel(..., input_dep_pi=True, momentum=0.9)

# 전체 (conv + sel + mom + prior_bias)
m = PRISMLangModel(..., use_conv=True, input_dep_pi=True, momentum=0.9, prior_bias=True)
```

### Local Conv1d: use_conv

Mamba의 causal depthwise conv1d를 에너지 관측 전처리에 추가.
char LM에서 "th"→"e", "ing", "tion" 같은 로컬 n-gram 패턴을 효율적으로 포착.

- 학습: vectorized `Conv1d(emb_dim, emb_dim, k=4, groups=emb_dim)` — 전체 시퀀스 일괄처리
- 생성: 순차 conv buffer — 마지막 d_conv 토큰 유지
- 추가 파라미터: `emb_dim × d_conv + emb_dim = 64 × 4 + 64 = 320` (매우 저렴)

### Z-Gate: use_gate

Mamba의 `y × SiLU(z)` 게이팅 메커니즘의 PRISM 유사체.

```
x = x × SiLU(W_gate · u_raw)  — 입력이 상태의 어떤 차원을 열고/닫을지 선택
```

- `gate_proj`: `Linear(emb_dim, d)`, bias initialized to 1.278 → SiLU(1.278) ≈ 1.0 (초기 identity)
- K-step 이후, RMS normalize 이전에 적용
- 추가 파라미터: `emb_dim × d + d = 64 × 168 + 168 = 10,920`

### N-gram Bypass: use_bypass

에너지 상태 독립적인 n-gram 단축로. 로컬 패턴을 직접 예측 분포에 기여.

```
logits = output_proj(norm_f(x)) + bypass_proj(u_all)  — u_all은 conv처리된 임베딩
```

- `bypass_proj`: `Linear(emb_dim, vocab_size)`, zeros init → 학습 전 효과 없음
- use_conv=True와 함께 사용 시 4-gram 컨텍스트를 직접 어휘 분포에 매핑
- 추가 파라미터: `emb_dim × vocab_size = 64 × 65 = 4,160`

---

## de-risking 단계

### 완료
- [x] **Stage 1** — 에너지 수렴 확인: dx/ds = −∂E/∂x 가 K번 반복 후 수렴
- [x] **Stage 2** — char LM 기본 동작: PRISM이 TinyShakespeare에서 학습 가능
- [x] **Stage 3** — 비대칭 Hebbian 복원: key=x̂, value=ê_mem → K-effect 0.04→1.74 (43×)
- [x] **Stage 4** — 설계 정합 검증: carry gate 제거, prior 없이 K-effect 없음 확인
- [x] **Stage 5/6** — u_rec 비선형화: 관측 증강 f([u_raw, x_prev]) 도입
- [x] **Stage 7** — Prior 항 도입: ½‖x−μ(x_prev)‖²_Π3 → K-effect +1.765 ppl (K2→K4)
- [x] **Stage 8** — 파라미터 매칭 완료: PRISM-K4 **12.942** vs LSTM **14.164** (−1.222 ppl), K-effect +0.659 ppl
- [x] **Stage 9** — Mamba 비교: Mamba **6.605** vs PRISM-K4 **14.525** (Mamba 승, +7.9 ppl); K-effect +0.951 ppl 유지
- [x] **Stage 9b** — 멀티모달 구현: 에너지에 시각 항 추가 → 이미지 없을 때 3.808 vs 있을 때 **3.218** ppl (+0.590 개선)
- [x] **Stage 10** — 단순화 분석: identity prior의 K-effect 3.5× 더 큼; u_rec 필수 확인; 10 epoch v1-K4 **12.794** vs slim-K4 15.048
- [x] **Stage 12** — Mamba 격차 원인 분석 (3 epoch 조기 신호): K=8 +0.129 ppl, rank=48 +0.007 ppl; Mamba **7.850** vs PRISM **28.381** (gap +20.5 ppl at 3 epochs); Mamba epoch 1부터 9.381 → conv1d가 핵심

### 진행 예정
- [ ] **Stage 11** — 적응형 K(t) 실측: 엔트로피 기반 K 선택 vs 고정 K 비교
- [ ] **Stage 13** — Selective PRISM 검증: Π(u) vs const Π, 파라미터 매칭 비교
- [ ] **Stage 14** — 전체 개선사항 ablation (실행 중): conv1d + input_dep_pi + momentum + prior_bias
- [ ] **Stage V** — GPU 스케일업: 50~150M params (V100)
