import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
import json

BLUE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"

def style(ax, xlab=None, ylab=None, title=None, logx=False):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    if xlab: ax.set_xlabel(xlab, color=INK2, fontsize=10)
    if ylab: ax.set_ylabel(ylab, color=INK2, fontsize=10)
    if title: ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    if logx: ax.set_xscale("log")

r3 = json.load(open("results/round3/results.json"))["experiments"]
e1 = r3["e1_grokking"]

h = e1["mul"]; s = h["step"]
fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2, 2]})
fig.patch.set_facecolor(SURF)
ax = axes[0]
ax.plot(s, h["train_acc"], color=BLUE, lw=2)
ax.plot(s, h["val_acc"], color=RED, lw=2)
ax.text(2600, 0.93, "train", color=BLUE, fontsize=10, fontweight="bold")
ax.text(23000, 0.42, "val", color=RED, fontsize=10, fontweight="bold")
ax.axvspan(18750, 36250, color=YELLOW, alpha=0.10, lw=0)
ax.text(25500, 0.06, "상전이 구간\n(val 10%→90%)", color=INK2, fontsize=8.5, ha="center")
ax.annotate("암기 완료 (train 99%)", xy=(21250, 1.0), xytext=(3500, 0.78),
            color=INK2, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
style(ax, ylab="accuracy", title="PRISM grokking 해부 — mod-97 곱셈 (d=512, K=8, wd=1.0)", logx=True)
ax = axes[1]
ax.plot(s, h["val_loss"], color=VIOLET, lw=2)
ax.set_yscale("log")
ax.set_yticks([0.3, 1, 3, 6])
ax.set_yticklabels(["0.3", "1", "3", "6"])
ax.minorticks_off()
ax.axvspan(18750, 36250, color=YELLOW, alpha=0.10, lw=0)
ax.annotate("암기 중 val loss 상승\n(과적합 심화)", xy=(15000, 5.6), xytext=(1800, 1.1),
            color=INK2, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
style(ax, ylab="val loss (log)", title="연속 지표도 같은 상전이 — 신기루 아님", logx=True)
ax = axes[2]
ax.plot(s, h["eta"], color=AQUA, lw=2)
ax.axvspan(18750, 36250, color=YELLOW, alpha=0.10, lw=0)
ax.annotate("상전이 후 완만한 감소", xy=(45000, 0.62), xytext=(6000, 0.30),
            color=INK2, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
style(ax, xlab="training step (log)", ylab="eta (기억 쓰기 강도)",
      title="기억 회로 강도: 학습과 함께 성장, 일반화 후 소폭 이완", logx=True)
fig.tight_layout()
fig.savefig("analysis_f1_grokking_anatomy.png", dpi=150, facecolor=SURF)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
fig.patch.set_facecolor(SURF)
ht = e1["transformer_add"]
ax.plot(ht["step"], ht["val_acc"], color=BLUE, lw=2)
ax.plot(e1["mul"]["step"], e1["mul"]["val_acc"], color=RED, lw=2)
ax.plot(e1["add"]["step"], e1["add"]["val_acc"], color=YELLOW, lw=2)
ax.text(1600, 0.9, "transformer (add)", color=BLUE, fontsize=9.5, fontweight="bold")
ax.text(30000, 0.99, "PRISM (mul)", color=RED, fontsize=9.5, fontweight="bold")
ax.text(29000, 0.58, "PRISM (add)", color="#c98500", fontsize=9.5, fontweight="bold")
style(ax, xlab="training step (log)", ylab="val accuracy",
      title="같은 상전이, ~25배 늦게 — val 정확도 (mod-97)", logx=True)
fig.tight_layout()
fig.savefig("analysis_f2_prism_vs_transformer.png", dpi=150, facecolor=SURF)
plt.close(fig)

cap = r3["e4_memory"]["capacity"]
pairs = [2, 4, 8, 16]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
fig.patch.set_facecolor(SURF)
colors = {32: AQUA, 64: BLUE, 128: VIOLET}
ax = axes[0]
for d in [32, 64, 128]:
    acc = [cap[f"d{d}_p{p}"]["final_val_acc"] for p in pairs]
    ax.plot(pairs, acc, "o-", color=colors[d], lw=2, ms=5)
offs = {32: -0.012, 64: 0.0, 128: 0.012}
for d in [32, 64, 128]:
    ax.text(16*1.12, cap[f"d{d}_p16"]["final_val_acc"] + offs[d], f"d={d}",
            color=colors[d], fontsize=9, va="center")
ax.set_xscale("log", base=2); ax.set_xticks(pairs); ax.set_xticklabels(pairs)
ax.axhline(1/63, color=INK2, lw=1, ls=":")
ax.text(2, 1/63+0.008, "chance", color=INK2, fontsize=8)
style(ax, xlab="저장 쌍 수 n_pairs", ylab="recall 정확도", title="정확도 ∝ 1/n_pairs — 폭 d를 키워도 거의 불변")
ax = axes[1]
for d in [32, 64, 128]:
    prod = [cap[f"d{d}_p{p}"]["final_val_acc"]*p for p in pairs]
    ax.plot(pairs, prod, "o-", color=colors[d], lw=2, ms=5)
ax.set_xscale("log", base=2); ax.set_xticks(pairs); ax.set_xticklabels(pairs)
ax.set_ylim(0, 1.05)
ax.axhline(0.76, color=INK2, lw=1, ls=":")
ax.text(2, 0.80, "acc × n_pairs ≈ 0.76", color=INK2, fontsize=8.5)
style(ax, xlab="저장 쌍 수 n_pairs", ylab="정확도 × n_pairs", title='"최근 ~1쌍만 기억" 시그니처 (일정한 곱)')
fig.tight_layout()
fig.savefig("analysis_f3_recall_recency.png", dpi=150, facecolor=SURF)
plt.close(fig)

ek = r3["e3_thinking_depth"]["eval_K"]
Ks = sorted(int(k.split("_")[1]) for k in ek)
accs = [ek[f"K_{K}"]["val_acc"] for K in Ks]
fig, ax = plt.subplots(figsize=(7.0, 4.0))
fig.patch.set_facecolor(SURF)
ax.plot(Ks, accs, "o-", color=BLUE, lw=2, ms=6)
ax.set_xscale("log", base=2); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
ax.annotate("학습 K=8에서만 능력 존재\n(step 재조정으로도 복구 불가)", xy=(8, 0.848),
            xytext=(15, 0.62), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
style(ax, xlab="eval K — 사고 깊이 (log2)", ylab="val accuracy",
      title="K-취성: 평형 해가 아니라 고정 깊이 회로 (grokked add 모델)")
fig.tight_layout()
fig.savefig("analysis_f4_k_brittleness.png", dpi=150, facecolor=SURF)
plt.close(fig)
print("4 figures saved with Korean font")
