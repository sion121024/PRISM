"""Fetch Kaggle kernel output and summarize emergence results.

Usage:  python fetch_and_summarize.py [--no-fetch]
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "kaggle_out"
KERNEL = "sion1210/prism-emergence"


def fetch():
    OUT.mkdir(exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", str(OUT)],
                   check=True)


def fmt_stats(s):
    if not s:
        return "-"
    return (f"10%@{s.get('step_10pct')}  90%@{s.get('step_90pct')}  "
            f"width={s.get('transition_width')}  final={s.get('final_acc')}")


def main():
    if "--no-fetch" not in sys.argv:
        fetch()
    r = json.loads((OUT / "results.json").read_text())
    ex = r["experiments"]
    print(f"env: {r['env']}")
    print(f"fast_weights: {r.get('use_fast_weights')}\n")

    if "e1_grokking" in ex:
        print("== E1 grokking ==")
        for k, h in ex["e1_grokking"].items():
            if "step" not in h:
                print(f"  {k}: {h}")
                continue
            print(f"  {k}: train {fmt_stats(h.get('stats_train'))}")
            print(f"  {k}: val   {fmt_stats(h.get('stats_val'))}")
            tr = h.get("stats_train", {}).get("step_90pct")
            va = h.get("stats_val", {}).get("step_90pct")
            if tr and va:
                print(f"  {k}: GROKKING DELAY = {va - tr} steps "
                      f"(x{va / max(tr, 1):.1f})")

    if "e2_scaling" in ex:
        print("\n== E2 scaling ==")
        for k in sorted(ex["e2_scaling"],
                        key=lambda s: int(s.split("_")[1])):
            v = ex["e2_scaling"][k]
            if "final_val_acc" not in v:
                print(f"  {k}: {v}")
                continue
            def f(x, spec=".3f"):
                return "None" if x is None else format(x, spec)
            print(f"  {k:6s} params={v.get('params')}  "
                  f"train={f(v.get('final_train_acc'))}  "
                  f"val={f(v.get('final_val_acc'))}  "
                  f"loss={f(v.get('final_val_loss'), '.4f')}")

    if "e3_thinking_depth" in ex:
        print("\n== E3 thinking depth ==")
        for k, v in ex["e3_thinking_depth"].get("eval_K", {}).items():
            print(f"  eval {k:5s} acc={v['val_acc']:.3f} "
                  f"loss={v['val_loss']:.4f}")
        for k, v in ex["e3_thinking_depth"].get("train_K", {}).items():
            if v.get("final_val_acc") is not None:
                print(f"  train {k:5s} train={v['final_train_acc']:.3f} "
                      f"val={v['final_val_acc']:.3f}  "
                      f"{fmt_stats(v.get('stats_val'))}")
            else:
                print(f"  train {k}: {v}")

    if "e4_memory" in ex:
        print("\n== E4 memory ==")
        for k, h in ex["e4_memory"].get("curves", {}).items():
            if "step" in h:
                print(f"  {k}: {fmt_stats(h.get('stats_val'))}  "
                      f"eta {h['eta'][0]:.3f}->{h['eta'][-1]:.3f}  "
                      f"chance={h.get('chance'):.3f}")
            else:
                print(f"  {k}: {h}")
        cap = ex["e4_memory"].get("capacity", {})
        if cap:
            print("  capacity (final val acc):")
            for k, v in cap.items():
                print(f"    {k:10s} acc={v.get('final_val_acc')}  "
                      f"eta={v.get('final_eta')}")

    if "e5_weight_decay" in ex:
        print("\n== E5 weight decay ==")
        for k, h in ex["e5_weight_decay"].items():
            if "step" in h:
                print(f"  {k}: val {fmt_stats(h.get('stats_val'))}")
            else:
                print(f"  {k}: {h}")


if __name__ == "__main__":
    main()
