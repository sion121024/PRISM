# PRISM — Predictive Recurrent Implicit State Machine

## 핵심 아이디어
"지각·기억·추론·행동은 전부 같은 에너지 함수 E를 경사하강하는 것."

고정크기 연속상태 x ∈ ℝ^d 가 에너지 E(x)를 내려감으로써 사고를 구현.

```
E(x) = ½‖u − Dx‖²_Π1 + ½‖(I−M)x‖²_Π2 + ½λ‖x‖²
dx/ds = −∂E/∂x = DᵀΠ1·εin − (I−M)ᵀΠ2·εmem − λx
ΔM    = η·(εmem·xᵀ) − γ·M
```

**이중시계:**
- 외부시계 t: 토큰/프레임마다 1틱, 추론 O(1)
- 내부시계 s: 틱마다 E 하강 K(t)번 (적응적), 빠른반응↔깊은사고

## 디렉토리 구조

```
prism/
  cell.py        # PRISMCell — 에너지 함수, 그래디언트, K-step 반복
  deq.py         # DEQ 솔버 (Anderson acceleration + implicit diff)
  model.py       # PRISMLangModel — 전체 언어 모델
tasks/
  copy_task.py   # Copy sequence task
  assoc_recall.py  # Associative recall task
  char_lm.py     # Character-level LM data
baselines/
  lstm_lm.py     # LSTM 비교 베이스라인
train.py         # 학습 스크립트
verify_convergence.py  # Stage 1 사활 검증: 에너지 수렴 확인
```

## 빠른 실행

```bash
# Stage 1 사활 검증 (에너지 수렴)
python verify_convergence.py

# Copy task 학습
python train.py --task copy --epochs 20 --K 4 --d 128

# Associative recall 학습 (K=3, CPU ~5초/epoch)
python train.py --task assoc_recall --epochs 20 --n_pairs 4 --K 3 --d 128 --emb_dim 32

# 문자 LM (PRISM, CPU ~1.6분/epoch)
python train.py --task char_lm --epochs 20 --K 2 --block_size 128

# 문자 LM (LSTM 베이스라인)
python train.py --task char_lm --epochs 20 --baseline lstm
```

## 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `d` | 256 | 상태 차원 |
| `emb_dim` | 64 | 임베딩/입력 차원 |
| `K` | 4 | 내부 반복 횟수 |
| `alpha` | 0.05 | 내부 스텝 크기 |
| `lam` | 0.01 | 상태 정규화 λ |
| `mem_eta` | 0.01 | 빠른가중치 학습률 η |
| `mem_gamma` | 0.001 | 빠른가중치 감쇠 γ |
| `simple_prior` | False | Identity prior μ=x_prev (0 params) |
| `prior_bias` | False | Biased prior μ=x_prev+b (+d params, 빠른 수렴) |
| `use_urec` | True | 관측 증강 ũ=f([u, x_prev]) (필수) |
| `input_dep_pi` | False | 선택적 precision Π1(u),Π2(u) ≈ Mamba B/C |
| `momentum` | 0.0 | K-step Heavy-ball β (0.9 권장) |
| `use_conv` | False | Depthwise conv1d n-gram 패턴 캡처 (Mamba 유사체, +320 params) |
| `d_conv` | 4 | conv1d 커널 크기 |
| `use_gate` | False | readout-only 게이트: h=norm(x)×SiLU(W_g·u) → logits. 재귀 상태 x 불변 (+10,920 params) |
| `use_compile` | False | torch.compile로 cell.iterate 퓨전 — 약 2× 속도 향상 (PyTorch 2.0+) |
| `use_deq` | False | DEQ implicit diff — O(1) 메모리 역전파 (Anderson acceleration) |
| `approximate_grad` | False | K-1 no_grad + 1 grad — 역전파 그래프 1/K로 축소 |
| `tbptt_window` | 0 | Truncated BPTT 윈도우 크기 (0=끔, 예: 16) — 긴 시퀀스 학습 시 메모리 절약 |
| `n_layers` | 1 | 계층적 예측 코딩 레이어 수. l>0 레이어는 x_{l-1}을 관측값으로 받음 |

## CPU 성능 최적화

train.py는 `torch.set_num_threads(1)` 을 자동으로 설정.
PyTorch 멀티스레드 오버헤드(소형 텐서 문제): 4 threads = 14ms, 1 thread = 0.1ms (136× 차이).

| 설정 | 속도 (B=64, T=64) |
|------|------------|
| slim-K4 base | ~780ms/batch |
| slim-K4 + use_compile | ~370ms/batch (~2.1× 빠름) |
| (참고) assoc_recall K=3 d=128 T=9 | ~30ms/batch |

## de-risking 단계

- [x] **Stage 1**: 에너지 수렴 + implicit diff 동작 확인
- [ ] **Stage 2**: 문자 LM ppl이 SSM 베이스라인과 경쟁
- [ ] **Stage 3**: 이중시계·적응K로 추론 태스크 ("더 생각하면 더 맞나")
- [ ] **Stage 4**: V100에서 50~150M 스케일업
- [ ] **Stage 5**: 비전 어댑터 + 행동 슬롯

## 학습 전략

느린가중치 θ (D, Π1, Π2, embed, unembed):
- K-step unroll로 backprop (Stage 1)
- DEQ implicit diff로 O(1) 메모리 backprop (Stage 2+)

빠른가중치 M:
- Hebbian 갱신 ΔM = η(εmem·xᵀ) − γM
- backprop 없음 (gradient detach)
