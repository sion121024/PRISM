# PRISM — Predictive Recurrent Implicit State Machine

> **베팅 프로젝트**: "지각·기억·추론·행동은 전부 *같은 에너지 함수 E를 경사하강*하는 것."

## 핵심 원리

```
상태 x∈ℝ^d, 빠른가중치 M (테스트타임 갱신), 느린가중치 θ (학습)

E(x) = ½‖u − D·x‖²_Π1  +  ½‖(I−M)·x‖²_Π2  +  ½λ‖x‖²
          ↑ 지각오차               ↑ 기억오차

dx/ds = −∂E/∂x   (K번 반복 = 사고의 깊이)
ΔM    = η·(ε_mem·xᵀ) − γ·M   (테스트타임 기억)
```

| 역할 | 메커니즘 |
|------|---------|
| 지각 | 예측오차(εin) 하강 |
| 추론 | K 크게 → 더 많은 내부 반복 |
| 실시간 반응 | K 작게 → 빠른 수렴 |
| 기억 학습 | 평형 x*를 M에 Hebbian으로 저장 |
| 멀티모달 | 모든 modality → 어댑터 → 같은 x |

**이중시계:**  외부 t (토큰/프레임, O(1) 고정상태) ⊥ 내부 s (틱마다 K(t)번 하강, 적응적 halting)

**학습:** BPTT 대신 DEQ(Deep Equilibrium) implicit 미분 → 메모리 O(1)

---

## 단계

| 단계 | 목표 | 상태 |
|------|------|------|
| 1 | 장난감: d=256, 문자LM, 에너지수렴+DEQ 확인 | ✓ 완료 |
| 2 | 문자LM ppl vs SSM 베이스라인 |  |
| 3 | 이중시계·적응K → 추론 task | **진행 중** |
| 4 | V100 텐서코어 가속, 50~150M 스케일업 |  |
| 5 | 비전 어댑터 + 행동 슬롯 (실시간 에이전트) |  |

### 창발 실험 (2026-07, Kaggle P100)

**mod-97 산술에서 grokking 달성** (d=512, val 94.1%, 지연 16.8k steps).
전제: 에너지의 비선형화(`u − D·tanh(x)`) — 2차 에너지는 상태 갱신이
선형이라 비선형 과제를 원리적으로 표현 불가(round 1 발견).
발견·교훈·병렬화 트릴레마 분석: [`experiments/emergence/FINDINGS.md`](experiments/emergence/FINDINGS.md)

---

## 빠른 시작

```bash
pip install torch

# 1. 아키텍처 sanity check (훈련 전 필수)
python probe_energy.py

# 2. 훈련 (tiny Shakespeare 자동 다운로드)
python train.py --K 8 --d 256 --epochs 20

# 3. 더 깊은 사고 (K=32) vs 빠른 반응 (K=4) 비교
python train.py --K 8 --K_eval 32 --compare --epochs 20

# 4. 생성
python generate.py --ckpt checkpoints/prism.pt --prompt "HAMLET:" --K 32
```

---

## 실패 위험 (베팅의 본질)

1. 에너지 하강 수렴 실패 (발산/진동)
2. EBM·implicit 미분 불안정
3. 멀티모달 binding 난제
4. 소규모서 트랜스포머 대비 품질 미달

→ `probe_energy.py`가 1번을 빠르게 검증. 실패하면 step 크기·정밀도(Π) 튜닝.

---

## 파일 구조

```
prism/
  __init__.py
  energy.py    ← PRISMCell: E, ∂E/∂x, descend(), update_M()
  model.py     ← PRISMLM: embed → cell → head, DEQ backward
  data.py      ← CharDataset
train.py       ← 훈련 루프
probe_energy.py ← sanity checks
generate.py    ← 텍스트 생성
```
