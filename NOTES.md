# PRISM — 세션 노트 (2026-06-16)

## 핵심 설계 결정

### 중심 명제
지각·기억·추론·행동은 전부 **같은 에너지 함수 E를 경사하강**하는 것.

### 수식
```
E(x) = ½‖u − D·x‖²_Π1  +  ½‖(I−M)·x‖²_Π2  +  ½λ‖x‖²
dx/ds = −∂E/∂x   (K번 반복 = 사고의 깊이)
ΔM    = η·(ε_mem·xᵀ) − γ·M   (테스트타임 기억)
```

### 이중시계
- **외부 t**: 토큰/프레임 단위, O(1) 고정 상태
- **내부 s**: 틱마다 K(t)번 하강, 적응적 halting
- 같은 모델, K만 다르게 → 빠른 반응(K=4) / 깊은 사고(K=32)

---

## Stage-1 실험 결과 (2026-06-16)

### Sanity check (probe_energy.py) — ALL PASS
| 테스트 | 결과 |
|--------|------|
| 에너지 수렴 | E: 25.0 → 20.3 ✓ |
| DEQ gradient | NaN 없음 ✓ |
| fast-weight M | 기억오차 0.39 → 0.08 ✓ |

### 문자 LM 훈련 (tiny Shakespeare 50KB, d=128, K=4, no DEQ)
| Epoch | val_ppl (K=8) | val_ppl (K=4) |
|-------|--------------|--------------|
| 1 | 21.50 | 21.96 |
| 3 | 11.41 | 10.80 |
| 7 | 10.50 | 8.91 |

→ 학습 자체는 잘 됨. K=8 vs K=4 차이는 현재 스케일에서 노이즈 범위.

---

## 환경 관련

- **Claude Code on the web**: CPU 4코어 (Xeon 2.8GHz), RAM 16GB, GPU 없음
- DEQ ON vs OFF: **88배 속도 차이** (CPU 환경에서는 DEQ 끄고 실험)
- 실제 훈련은 로컬 GPU 또는 Colab/Vast.ai 필요

---

## 발견된 문제 & 분석

### EBM 불안정성
PRISM은 일반 EBM보다 안정적 (negative sampling 불필요, 예측오차가 에너지).
DEQ는 Jacobian 스펙트럼 반경 < 1 조건 필요 → 스케일업 시 주요 위험.

### K 루프 효율
트랜스포머는 matmul 한 번, PRISM은 K번 순차 루프.
같은 파라미터 수여도 K배 느림 → GPU 커스텀 커널 필요 (스케일업 시).

---

## 다음 재설계 방향 (논의됨)

목표: **PRISM 근본 유지, 더 적은 파라미터로 더 똑똑하고 빠르게**

### 1. Anderson Acceleration (우선순위 높음)
K=8~20 gradient descent → K=3~5로 같은 수렴
- 구현 난이도: 낮음
- 기대 효과: 3~5x 속도 향상

### 2. 슬롯 구조 (표현력 향상)
`x ∈ ℝ^d` → `x_slots ∈ ℝ^{N×d}` (N=8)
- 파라미터 그대로, 표현력 N배
- 트랜스포머 multi-head의 에너지 버전

### 3. Low-rank Fast Weight
`M ∈ ℝ^{d×d}` → `M = A·Bᵀ` (r≪d)
- 메모리/연산 O(d²) → O(dr)
- 대용량 연상기억에 필수

---

## de-risking 단계 (설계서 기준)

| 단계 | 목표 | 상태 |
|------|------|------|
| 1 | 에너지 수렴 + DEQ 동작 확인 | ✓ 완료 |
| 2 | 문자LM ppl vs SSM 베이스라인 | 진행 중 |
| 3 | 이중시계·적응K로 추론 task | 대기 |
| 4 | V100 텐서코어 가속, 50~150M | 대기 |
| 5 | 비전 어댑터 + 행동 슬롯 | 대기 |

---

# 창발(Emergence) 실험 — Kaggle GPU 세션 (2026-07-10 ~ 11)

브랜치: `claude/transformer-singularity-theory-auafm5`
코드: `experiments/emergence/kaggle_emergence.py` (자체완결 Kaggle 커널)
환경: Kaggle P100 — 최신 torch 휠이 sm_60 미지원 → 부트스트랩이 CUDA 연산
실패를 감지하고 torch==2.4.1+cu121로 자동 재설치 (매 라운드 정상 작동)

## 실험 축 (여러가지 창발 축)
| 축 | 실험 | 질문 |
|----|------|------|
| 학습 시간 | E1 grokking (mod-97) | 암기 후 일반화가 갑자기 오는가 |
| 규모 d | E2 스케일 스윕 | 임계 폭이 존재하는가 (+신기루 검증) |
| 사고 깊이 K | E3/E6 | 내부 하강 반복이 능력 축이 되는가 |
| 기억 용량 | E4 연상 recall | fast weight 회로가 형성되는가 |
| 정규화 | E5 wd 스윕 | 창발에 wd가 필요한가 |

## Round 1 — 선형성 결함 발견
- 트랜스포머 베이스라인: 교과서적 grokking (train 90%@500, val 90%@1400)
- PRISM: 60k steps에도 train 13% — 암기조차 실패
- **진단**: E가 x에 대해 2차식 → ∂E/∂x 선형 → 토큰당 상태 갱신이 아핀 변환.
  비선형 과제(mod 연산)를 원리적으로 표현 불가. M의 쌍선형성만으론 부족.

## Round 2 — 비선형 사전 도입
- 수정: E = ½‖u − D·tanh(x)‖²_Π1 + … (에너지 하강 원리는 유지)
- preflight(mod-23): 비선형 99.7% vs 선형 89% — 결함 수정 확인
- mod-97 mul에서 첫 일반화 신호 (val 11% ≫ chance 1%)
- recall(d32, 2쌍): 3.4% → 39.1% — 기억 회로 형성 시작
- 새 문제: lr=1e-2가 d≥256에서 발산, softplus eta가 1.15까지 폭주

## Round 3 — GROKKING 달성 (d=512)
- 수정: fast weight를 외적 이력으로 정확 분해 (B×d×d 비실체화, dense와
  일치 자기검증 포함) → d=512 가능. eta sigmoid 유계화. 폭 스케일 lr
  (lr ∝ 1/√d) + 발산 시 반감 재시도.
- **mod-97 mul: 완전한 grokking** — train 90%@19.5k, val 90%@36.3k,
  지연 16.8k steps, 최종 val 94.1%
- mod-97 add: train 100%, val 84.8% (50k 예산 내 상전이 진행 중)
- 트랜스포머 대비 grokking ~25배 느림 (지연 1k vs 16.8k)
- **eval-K 취성 발견**: 능력이 학습 K=8에서만 존재 (84.8% vs 그 외 ~1%,
  K≥16은 loss NaN). step 재조정(K·step 불변)으로도 복구 안 됨 →
  모델이 에너지 평형이 아니라 "고정 깊이 전개 회로"를 학습한 것.
  DEQ적 anytime 추론 주장에 대한 중대한 반례.

## Round 4 (실행 중)
- E2 스케일 스윕 d∈{32,…,512} @30k — 임계 폭 곡선
- E6 (신규): K~U{4,16} 무작위 학습 → K-불변 해 강제, eval-K 일반화 +
  "더 오래 생각하면 더 정확한가" (진짜 test-time compute 창발) 검증
- E3 train-K, E5 wd (d=256)

## 결과 아카이브
- `experiments/emergence/results/round{1,2,3}/` — results.json, 플롯,
  round3는 grokked 체크포인트(e1_add.pt, e1_mul.pt) 포함
