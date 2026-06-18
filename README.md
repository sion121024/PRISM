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
```

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

### 진행 예정
- [ ] **Stage 11** — 적응형 K(t) 실측: 엔트로피 기반 K 선택 vs 고정 K 비교
- [ ] **Stage 12** — Mamba 격차 분석 진행 중: K=8 빠른 수렴 확인 (K4 대비 4× 빠른 개선)
- [ ] **Stage 13** — Selective PRISM 검증: Π(u) vs const Π, 파라미터 매칭 비교
- [ ] **Stage 14** — 전체 개선사항 ablation: input_dep_pi + momentum + norm_f + prior_bias
- [ ] **Stage V** — GPU 스케일업: 50~150M params (V100)
