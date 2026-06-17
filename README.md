# PRISM — Predictive Recurrent Implicit State Machine

> "지각·기억·추론·행동은 전부 같은 에너지 함수 E를 경사하강하는 것."

고정 크기 연속 상태 x ∈ ℝ^d가 에너지 E(x)를 내려감으로써 사고를 구현하는 순환 신경망 아키텍처.

---

## 핵심 아이디어

```
E(x) = ½‖u − g(x)‖²_Π1       ← 지각: 현재 입력 설명
     + ½‖(I−M)x‖²_Π2          ← 기억: Hebbian fast-weight와 일치
     + ½‖x − μ(x_prev)‖²_Π3   ← 추론: 이전 상태로부터 예측
     + ½λ‖x‖²                  ← 정규화

dx/ds = −∂E/∂x   (K번 반복 = 내부 사고)
ΔM    = η(εmem ⊗ x) − γM   (비대칭 Hebbian)
```

### 이중 시계 (Dual Clock)
| 시계 | 틱 | 역할 |
|------|-----|------|
| 외부 t | 토큰마다 1회 | 입력 처리, Hebbian 갱신 |
| 내부 s | 토큰마다 K회 | 에너지 하강 = 사고 |

**K가 클수록 더 깊은 추론** — 어려운 토큰에 더 많은 K 배분(적응형 K(t)) 가능.

---

## 설계 원칙

1. **모든 계산 = 에너지 하강**: carry gate, layer norm 등 에너지 밖 변환 없음
2. **비대칭 Hebbian**: key=normalize(x), value=normalize(εmem) → 오류신호 기반 연상기억
3. **Prior 항 필수**: μ(x_prev)가 없으면 K-effect 없음 (에너지 지형이 단순해짐)
4. **MLP decoder** g(x): 비볼록 E → K-step이 실제 추론을 수행

---

## 검증 결과 (TinyShakespeare, char-level LM)

### Stage 7: K-effect 증명 (116K params)

| 구성 | val ppl | params |
|------|---------|--------|
| O-prior-K2 | 16.715 | 116K |
| **P-prior-K4** | **14.951** | 116K |
| LSTM (참조) | 20.542 | 56K |

**K2→K4: 1.765 ppl 개선** — "더 많이 생각 = 더 똑똑" 실증.

### 설계 정합 ablation 요약

| 조건 | val ppl | 비고 |
|------|---------|------|
| linear decoder + carry gate | 27.1 | Stage 3-C |
| MLP decoder + carry gate, K=2 | 24.7 | Stage 3-D |
| MLP decoder + carry gate, K=4 | 22.9 | Stage 3-E, K-effect 1.7↑ |
| MLP decoder, no carry, no prior, K=2~8 | ~28.3 | Stage 4 (K-effect 없음) |
| MLP decoder + prior, K=2 | 16.7 | Stage 7-O |
| MLP decoder + prior, K=4 | **14.9** | Stage 7-P |

> Prior 항이 K-effect의 핵심. Prior 없이는 K=2,4,8 모두 ~28 ppl로 동일.

---

## 구조

```
prism/
  cell.py        # PRISMCell — 에너지 함수, 그래디언트, K-step, 적응형 K
  deq.py         # DEQ 솔버 (Anderson acceleration)
  model.py       # PRISMLangModel — 전체 언어 모델
tasks/
  copy_task.py
  assoc_recall.py
  char_lm.py     # TinyShakespeare char LM
baselines/
  lstm_lm.py
stage2_compare.py      # PRISM vs LSTM 기본 비교
stage3_ablation.py     # 비대칭 Hebbian + carry gate ablation
stage4_design.py       # 설계 정합 검증 (no carry, state_norm)
stage7_prior.py        # Prior 항: K-effect 증명
stage8_param_match.py  # 파라미터 매칭 최종 비교 (~54K)
```

---

## 빠른 실행

```bash
# 파라미터 매칭 최종 비교 (PRISM ~54K vs LSTM ~56K)
python stage8_param_match.py --epochs 10 --block_size 64

# K-effect 증명 (prior 항)
python stage7_prior.py --epochs 5 --block_size 64

# 설계 정합 ablation
python stage3_ablation.py --epochs 5 --block_size 64
```

---

## 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `d` | 256 | 상태 차원 |
| `emb_dim` | 64 | 임베딩 차원 |
| `K` | 4 | 내부 반복 횟수 (사고 깊이) |
| `alpha` | 0.05 | 내부 스텝 크기 |
| `mem_scale` | 4.0 | Hebbian 메모리 강도 |
| `mem_rank` | 32 | 슬라이딩 메모리 rank |
| `use_prior` | True | Prior 항 활성화 (K-effect 필수) |

---

## de-risking 단계

- [x] **Stage 1**: 에너지 수렴 확인
- [x] **Stage 2**: char LM 기본 동작
- [x] **Stage 3**: 비대칭 Hebbian + MLP decoder → K-effect 첫 확인
- [x] **Stage 4**: carry gate 없는 설계 정합 검증 (prior 없이 K-effect 없음 확인)
- [x] **Stage 7**: Prior 항 → K-effect 실증 (1.765 ppl, K2→K4)
- [ ] **Stage 8**: 파라미터 매칭 (~54K) 최종 비교
- [ ] **Stage 9**: 적응형 K(t) 검증 (어려운 토큰 = 더 많은 K)
- [ ] **Stage 10**: V100 스케일업 (50~150M)
