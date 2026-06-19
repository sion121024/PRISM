# PRISM — Predictive Recurrent Implicit State Machine

> "지각·기억·추론·행동은 전부 같은 에너지 함수 E를 경사하강하는 것."

고정 크기 연속 상태 x ∈ ℝ^d가 에너지 E(x)를 내려감으로써 사고를 구현하는 순환 신경망 아키텍처.

### 핵심 성과 (vs Mamba, 파라미터 매칭)

- 🏆 **추론 우위**: 연상 회상 n_pairs=12에서 PRISM이 **더 적은 파라미터로 Mamba 격파**
  (PRISM 0.151 vs Mamba 0.110). 난이도↑에서 Mamba 급락 vs PRISM 우아한 저하 —
  명시적 Hebbian 기억이 부하에 강건 (Stage 17/19b, 공정 매칭 확인).
- ⚡ **효율**: diag_scan으로 K-step 에너지 하강을 **O(1)** 계산 — K=64에서 **151× 가속**.
- 🔁 **이중시계**: 추론 시 K를 늘리면 정확도 향상 (Mamba에 없는 적응형 사고 깊이).
- 📊 char-LM은 Mamba 우위 (PRISM 7.58 vs 6.45) — 빠른 n-gram 포착이 핵심인 전장.
- 🖼️ **멀티모달**: 텍스트+시각+행동을 *같은 에너지 E의 항*으로 통합(fusion layer 불필요).
  결정 태스크에서 시각 항이 행동 정확도 +0.49 기여 (Full 0.71 vs No-Vision 0.23).
- 🌏 **이중언어**: 한국어+영어를 한 모델·한 상태에 동시 학습. 서브워드 12M 모델로
  **문장 수준 유창성** 달성 (영어 ppl 112, 한국어 ppl 295; Kaggle T4).
- ☁️ **GPU 스케일업**: Kaggle 2×T4에서 추론 우위가 스케일에서도 유지(PRISM 4/5 난이도 승, 147K params).

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
| `use_gate` | False | readout-only 게이트: h = norm(x)×SiLU(W_g·u) → logits. 재귀 상태 x 불변 (+10,920 params) |
| `n_layers` | 1 | 계층적 예측 코딩 레이어 수 (l>0은 x_{l-1}을 관측값으로 받음) |
| `diag_scan` | False | 대각 병렬 K-scan: 에너지 하강 O(1) 닫힌 형식 (Mamba parallel scan 동형) |

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

### Readout Gate: use_gate (설계철학 준수)

Mamba의 `y × SiLU(z)` 게이팅을 **출력 단계에만** 적용 — 재귀 상태는 순수 에너지 최솟값 유지.

```
x_t* = argmin_x E(x)            ← 에너지 하강으로 얻은 순수 상태 (게이트 영향 없음)
h    = norm_f(x_t*)
h    = h × SiLU(W_g · u)        ← readout 게이트: 어떤 차원을 출력에 쓸지 선택
logits = output_proj(h)
```

**왜 설계철학에 맞는가**: 게이트는 `logits = readout(x*)` 의 읽기 함수에만 작용.
재귀로 다음 토큰에 전달되는 `x_t*` 는 에너지 함수 E의 최솟값 그대로 — 사고 과정 불변.
(이전 버전은 `x = x × gate` 로 상태를 직접 수정했으나 설계 위반으로 제거됨.)

- `gate_proj`: `Linear(emb_dim, d)`, bias=1.278 → SiLU(1.278) ≈ 1.0 (초기 identity)
- 추가 파라미터: `emb_dim × d + d = 64 × 168 + 168 = 10,920`
- **실측 효과**: char_lm 5 epoch에서 baseline 24.18 → **13.14 ppl** (1.84× 개선, 단일 최대)

### 계층적 예측 코딩: n_layers (설계철학 확장)

에너지 함수를 다층으로 쌓는 정통 predictive coding 구조.

```
Layer 0: E_0(x_0) = ½‖ũ − g(x_0)‖²      ← 토큰 임베딩을 관측
Layer l: E_l(x_l) = ½‖x_{l-1} − g(x_l)‖² ← 하위 레이어 상태를 관측 (identity prior)
```

각 레이어는 독립 PRISMCell + 자체 Hebbian 기억 M_l. 하위 레이어 출력이 상위의 "관측값"이
되어, 상위 레이어가 더 추상적인 표현으로 에너지를 최소화. 모든 레이어가 동일한
`E 경사하강` 원칙을 따름 — 설계철학 그대로 깊이만 확장.

### 대각 병렬 K-scan: diag_scan (속도 최적화)

K번 순차 에너지 하강을 닫힌 형식으로 O(1) 계산. Mamba의 parallel scan과 동형이되,
**같은 에너지 함수를 최소화**하므로 설계철학 준수.

```
x_{k+1} = (I − αH) x_k + αb       ← 선형 재귀 (H = 대각 Hessian 근사)
x_K     = aᴷ x_0 + b·(aᴷ−1)/(a−1) ← 닫힌 형식 (a = 1 − αH)
```

순차 K-step과 최대오차 0.006 (norm 14 대비 0.04%) — 같은 최솟값에 수렴.

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
- [x] **Stage 3 (이중시계 검증)** — 훈련 K=4 고정 모델에서 **추론 K↑ → ppl↓** 단조 입증:
  K=1→28.67, K=2→28.42, K=4→28.29, K=8→28.25, K=16→28.24 (수렴 체감).
  같은 가중치로 추론 시 에너지를 더 깊이 하강할수록 정확 — **Mamba에 없는 PRISM 고유 능력**
  (학습 없이 추론비용↔정확도 trade-off). gate 모델도 동일 효과(15.10→14.98).
- [x] **Stage 14c (readout gate)** — gate가 단일 최대 개선: baseline **19.79** → gate **13.24** ppl
  (5 epoch, 1.49×). 재귀 상태 x*는 에너지 최솟값 유지 — 설계철학 준수.

### ✅ Stage 17 (추론 task): PRISM이 파라미터 대비 Mamba 격파

**핵심 결과** — 연상 회상(associative recall)에서 난이도가 오를수록 PRISM이 Mamba를 역전:

| 난이도 (n_pairs) | PRISM (20K) | Mamba (35K) | 승자 |
|------------------|-------------|-------------|------|
| 4 (쉬움) | 0.344 | 0.560 | Mamba |
| 8 (중간) | 0.231 | 0.254 | Mamba (접전) |
| **12 (어려움)** | **0.162** | 0.113 | **🏆 PRISM** |

난이도↑에서 **Mamba 급락**(0.560→0.113, −0.447) vs **PRISM 우아한 저하**(0.344→0.162, −0.182).
가장 어려운 난이도에서 PRISM이 **43% 적은 파라미터**(20K vs 35K)로 Mamba를 이김.

**왜**: PRISM의 명시적 content-addressable Hebbian 기억 M (½‖(I−M)x‖²_Π2)이 부하 증가에
강건. Mamba의 고정크기 압축 상태는 많은 key-value 쌍에서 정보 충돌로 붕괴.
**설계철학이 예측한 전장에서의 승리** — "기억·추론은 같은 에너지 E 하강".

### char-LM head-to-head (Mamba의 전장)

**파라미터 매칭 (TinyShakespeare, block=128):**

| 모델 | params | val ppl | 비고 |
|------|--------|---------|------|
| Mamba-d96 | 79,104 | 6.82 | 5 epoch 수렴 |
| Mamba-d128 | 130,048 | **6.45** | 5 epoch 수렴 |
| PRISM baseline | 79,552 | 19.79 | linear, K=4 |
| PRISM +gate | 96,192 | 13.24 | readout gate |
| PRISM +gate+conv+all | 117,568 | **7.58** (20ep 수렴) | Mamba 근접 (+1.13, 17%) |

char-LM은 빠른 n-gram 포착(Mamba conv1d 강점)을 보상 → PRISM이 근접하나 근소 열세
(PRISM 7.58 vs Mamba 6.45, params는 PRISM이 10% 적음).
PRISM의 우위는 **빠른 암기가 아닌 부하 하의 추론**에 있음 (Stage 17/19b).

### Stage 19b: 공정 파라미터 매칭으로 추론 승리 재확인

Stage 17은 Mamba가 75% 큰 상태(35K)였음 → 공정 매칭(Mamba-d48 21.8K vs PRISM 20.3K) 재실험:

| n_pairs | PRISM (20.3K) | Mamba (21.8K) | 승자 |
|---------|---------------|---------------|------|
| 4 | 0.334 | 0.542 | Mamba |
| 8 | 0.220 | 0.227 | Mamba (0.007) |
| **12** | **0.151** | 0.110 | **🏆 PRISM** |
| 16 | 0.086 | 0.090 | Mamba (0.004, 둘 다 붕괴) |

**공정 매칭에서도 PRISM이 더 적은 파라미터로 어려운 추론(n_pairs=12) 격파** — 승리 견고.
n_pairs=16은 이 규모(20K)에선 둘 다 랜덤 근처(≈0.0625)로 붕괴 → 무의미한 비교.
**PRISM의 우위는 "어렵지만 풀 수 있는" 영역**: Mamba 압축 상태는 붕괴하나
PRISM 명시적 Hebbian 기억은 버팀. Stage 17(Mamba 35K)·19b(공정 21.8K) 양쪽 확인.

### ⚡ diag_scan 효율화: K 무관 O(1) 에너지 하강

깊은 사고(큰 K)의 비용을 닫힌 형식으로 제거 (Mamba parallel scan 동형, 같은 E 최소화):

| K | 순차(ms) | diag(ms) | 속도이득 | 상대오차 |
|---|---------|----------|---------|---------|
| 4 | 1.587 | 0.158 | 10× | 0.2% |
| 16 | 5.555 | 0.186 | 30× | 0.4% |
| 64 | 23.357 | 0.155 | **151×** | 0.4% |

- [x] **Stage 17/19b** — 추론에서 PRISM이 파라미터 대비 Mamba 격파 (공정 매칭 재확인)
- [x] **Stage 16** — char-LM 수렴: gate+conv+all 20ep → 7.58 ppl (Mamba 6.45 근접)
- [x] **Stage 15** — 비볼록 MLP decoder ≈ linear (이 규모선 decoder 무차이)
- [x] **diag_scan** — K-step O(1) 닫힌 형식, 10~151× 가속 (상대오차 0.4%)

### ☁️ Stage 4 (GPU 스케일업, Kaggle 2×T4)

**추론 스케일업** (`stage21_reasoning_scaleup.py`) — PRISM d=512(147K) vs 공정매칭
Mamba-d136(140K), key/val_vocab=32:

| n_pairs | PRISM | Mamba | 승자 |
|---------|-------|-------|------|
| 8 (쉬움) | 0.199 | 0.316 | Mamba |
| 12 | 0.148 | 0.088 | 🏆 PRISM |
| 16 | 0.129 | 0.059 | 🏆 PRISM |
| 20 | 0.097 | 0.042 | 🏆 PRISM |
| 24 (어려움) | 0.089 | 0.039 | 🏆 PRISM |

**PRISM 4/5 승** — 부하↑ Mamba 급락(0.316→0.039) vs PRISM 우아한 저하(0.199→0.089).
추론 우위가 GPU 스케일에서도 유지. (이번 K-sweep은 단조 미재현 — 이 task는 K=1-2 수렴.)

**char-LM 스케일업** (`stage20_gpu_scaleup.py`) — PRISM 4M params가 enwik8에서
안정 학습(T4 5.4GB, 1378 tok/s). char-LM은 Mamba 우세(Stage 2와 일관).

**🌏 한국어+영어 LM** — 한·영을 같은 상태에 동시 학습:
- 바이트 단위(`stage22_bilingual.py`): 614K params, 토크나이저 없이(2.84/3.41 bpc) — 단어 조각 수준.
- **유창성**(`stage23_bilingual_fluent.py`): ByteLevel BPE 서브워드 + 12M params로
  **문장 수준 유창성**. 영어 Simple English Wikipedia로 깨끗한 산문(ppl 112),
  한국어 자연스러운 문장(ppl 295). 에너지 코어 불변, 입력만 바이트→서브워드.
  언어 태그(`<ko>`/`<en>`)로 코드스위칭 해결(v3 확인), `generate()` repetition_penalty 추가.

### 🖼️ Stage 5 (멀티모달 + 행동 슬롯)

`prism/agent.py` — 텍스트+시각+행동을 *같은 에너지 E의 항*으로 통합. 결정 토큰에서
행동 하강: **Full 0.71 vs No-Vision 0.23**(시각→행동 기여 +0.49). 시각 멀티모달은
ppl +0.59 개선. "지각·기억·추론·행동 = 같은 E 하강" 실증.

> 주의: 단일 T4 예산상 50~150M은 미도달(~mid scale 4~6M까지 검증). 추론·멀티모달·
> 이중언어로 설계철학의 핵심 주장(같은 E 하강)을 GPU에서 확인.
