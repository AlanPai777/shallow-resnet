"""Recompute every statistic quoted in report/report_en.md from the raw per-run JSON files.

Usage (from the repository root):
    python analysis/stats.py

Reads   results/s8_residual_depth_results.json          (constant learning rate; "v1" / first stage)
        results/s8_residual_depth_results_cosine.json   (cosine learning rate;  "v2" / primary protocol)
Needs only numpy. Prints plain text; nothing is written to disk.

Metrics
  final   accuracy after the last epoch (no epoch is selected)
  legacy  accuracy at the epoch with the lowest loss on the test set (selects on the test set)
"""
import json
from pathlib import Path

import numpy as np

from stats_utils import anova_2x2, cohens_d, permutation_p, unpooled_did, welch

RESULTS = Path(__file__).resolve().parent.parent / "results"
V1 = json.load(open(RESULTS / "s8_residual_depth_results.json"))
V2 = json.load(open(RESULTS / "s8_residual_depth_results_cosine.json"))


def runs(R, depth, residual):
    return sorted((r for r in R if r["depth"] == depth and r["use_residual"] == residual), key=lambda r: r["seed"])


def cells(R, metric):
    return {(d, s): np.array([metric(r) for r in runs(R, d, s)]) * 100 for d in (8, 20) for s in (True, False)}


final = lambda r: r["valid_accuracies"][-1]
legacy = lambda r: r["test_acc"]


def header(text):
    print(f"\n{'#' * 8} {text}")


def full_report(label, acc):
    header(label)
    for (d, s), a in acc.items():
        print(f"depth={d:2} residual={'ON ' if s else 'OFF'} seeds={np.round(a, 2)} mean={a.mean():.2f} sd(n-1)={a.std(ddof=1):.2f}")
    for d in (8, 20):
        w = welch(acc[(d, True)], acc[(d, False)])
        print(f"  effect at depth {d}: {w['diff']:+.2f}  95% CI [{w['lo']:+.2f}, {w['hi']:+.2f}]  Welch t={w['t']:.2f} df={w['df']:.2f} "
              f"p={w['p']:.4f}  exact-permutation p={permutation_p(acc[(d, True)], acc[(d, False)]):.2f}  "
              f"Cohen d={cohens_d(acc[(d, True)], acc[(d, False)]):+.2f}")
    A = anova_2x2(acc)
    U = unpooled_did(acc)
    print(f"  ANOVA (pooled): residual F(1,8)={A['F_res']:.2f} p={A['p_res']:.4f} | depth F={A['F_dep']:.2f} p={A['p_dep']:.2e} | "
          f"interaction F={A['F_int']:.2f} p={A['p_int']:.4f}")
    print(f"  difference-in-differences = {A['did']:+.2f}  pooled 95% CI [{A['did_lo']:+.2f}, {A['did_hi']:+.2f}]  cell variances {np.round(A['cell_vars'], 3)}")
    print(f"  unpooled difference-in-differences: t={U['t']:.2f} df={U['df']:.2f} p={U['p']:.4f} 95% CI [{U['lo']:+.2f}, {U['hi']:+.2f}]")
    return acc, A


# --------------------------------------------------------------------------------------------------
header("Data checks")
print("runs per file:", len(V1), len(V2), "| schedules:", {r.get("schedule", "constant") for r in V1}, {r["schedule"] for r in V2})
print("parameter counts (depth, params):", sorted({(r["depth"], r["num_params"]) for r in V2}), "identical for residual on/off:",
      all(len({r["num_params"] for r in V2 if r["depth"] == d}) == 1 for d in (8, 20)))
m1 = {(r["n"], r["use_residual"], r["seed"]): r for r in V1}
dl = max(max(abs(r["train_losses"][0] - m1[(r["n"], r["use_residual"], r["seed"])]["train_losses"][0]),
             abs(r["valid_losses"][0] - m1[(r["n"], r["use_residual"], r["seed"])]["valid_losses"][0])) for r in V2)
da = max(abs(r["valid_accuracies"][0] - m1[(r["n"], r["use_residual"], r["seed"])]["valid_accuracies"][0]) for r in V2)
print(f"epoch-1 identity across recipes (same seed): max |diff| loss = {dl}, accuracy = {da}  (0 means bit-identical)")
print("v2 runs whose lowest test loss is at the last epoch:", sum(r["best_epoch"] == 10 for r in V2), "/ 12;",
      "v1 runs with best epoch >= 9:", sum(r["best_epoch"] >= 9 for r in V1), "/ 12")

# --------------------------------------------------------------------------------------------------
a_v2f, A_v2f = full_report("Primary protocol: cosine LR, final epoch (Tables 2, 3)", cells(V2, final))
a_v1f, A_v1f = full_report("Constant LR, final epoch, no selection (Table 3)", cells(V1, final))
a_v1l, A_v1l = full_report("Constant LR, legacy best-test-loss epoch (Table 3)", cells(V1, legacy))

header("Table 3 summary (Residual ON - OFF, points)")
for name, acc, A in (("constant, legacy", a_v1l, A_v1l), ("constant, final", a_v1f, A_v1f), ("cosine, final", a_v2f, A_v2f)):
    w8, w20 = welch(acc[(8, True)], acc[(8, False)]), welch(acc[(20, True)], acc[(20, False)])
    print(f"{name:17s} depth 8 {w8['diff']:+.2f} [{w8['lo']:+.2f}, {w8['hi']:+.2f}] | depth 20 {w20['diff']:+.2f} [{w20['lo']:+.2f}, {w20['hi']:+.2f}] "
          f"| DiD {A['did']:+.2f}, pooled p={A['p_int']:.4f}")

header("Table 4: effect of annealing on final-epoch accuracy")
for d in (8, 20):
    for s in (True, False):
        b, c = a_v1f[(d, s)], a_v2f[(d, s)]
        print(f"depth={d:2} residual={'ON ' if s else 'OFF'}: constant {b.mean():.2f} ± {b.std(ddof=1):.2f} -> cosine {c.mean():.2f} ± {c.std(ddof=1):.2f} ({c.mean() - b.mean():+.2f})")

header("Legacy vs final accuracy, constant LR, depth 20, per run (points)")
for s in (True, False):
    for r in runs(V1, 20, s):
        print(f"residual={'ON ' if s else 'OFF'} seed {r['seed']}: legacy {r['test_acc'] * 100:.2f}  final {r['valid_accuracies'][-1] * 100:.2f}  "
              f"difference {(r['test_acc'] - r['valid_accuracies'][-1]) * 100:+.2f}")

header("Per-epoch mean validation-accuracy gap, ON - OFF (points)")
for name, R in (("constant", V1), ("cosine", V2)):
    for d in (8, 20):
        on = np.array([r["valid_accuracies"] for r in runs(R, d, True)]).mean(0) * 100
        off = np.array([r["valid_accuracies"] for r in runs(R, d, False)]).mean(0) * 100
        print(f"{name:8s} depth {d:2}: {np.round(on - off, 1)}")

header("Cosine runs: losses and train-validation gap at the final epoch")
for d in (8, 20):
    for s in (True, False):
        rr = runs(V2, d, s)
        tl = np.array([r["train_losses"][-1] for r in rr])
        vl = np.array([r["final_test_loss"] for r in rr])
        ta = np.mean([r["train_accuracies"][-1] for r in rr]) * 100
        va = np.mean([r["valid_accuracies"][-1] for r in rr]) * 100
        print(f"depth={d:2} residual={'ON ' if s else 'OFF'}: train loss {tl.mean():.3f}  val loss {vl.mean():.3f}  train-val accuracy gap {ta - va:+.2f}")
for d in (8, 20):
    for key, lab in (("train", "train loss"), ("val", "validation loss")):
        f = (lambda r: r["train_losses"][-1]) if key == "train" else (lambda r: r["final_test_loss"])
        a = np.array([f(r) for r in runs(V2, d, True)])
        b = np.array([f(r) for r in runs(V2, d, False)])
        w = welch(a, b)
        print(f"depth {d:2} final {lab} difference (ON - OFF): {w['diff']:+.3f}  95% CI [{w['lo']:+.3f}, {w['hi']:+.3f}]  p={w['p']:.4f}")

header("Epoch-to-epoch volatility of validation accuracy (mean |change|, points)")
for name, R in (("constant", V1), ("cosine", V2)):
    for d in (8, 20):
        for s in (True, False):
            va = np.array([r["valid_accuracies"] for r in runs(R, d, s)]) * 100
            print(f"{name:8s} depth={d:2} residual={'ON ' if s else 'OFF'}: all epochs {np.abs(np.diff(va, axis=1)).mean():.2f}   last 3 changes {np.abs(np.diff(va[:, -4:], axis=1)).mean():.2f}")

header("Other checkpoint rules on the constant-LR runs (mean over seeds)")
for label, fn in (("final epoch", lambda r: r["valid_accuracies"][-1]),
                  ("mean of epochs 8-10", lambda r: np.mean(r["valid_accuracies"][-3:])),
                  ("peak accuracy", lambda r: max(r["valid_accuracies"])),
                  ("legacy best-loss epoch", lambda r: r["test_acc"])):
    o = cells(V1, fn)
    print(f"{label:24s} depth 8: ON {o[(8, True)].mean():.2f} OFF {o[(8, False)].mean():.2f} ({o[(8, True)].mean() - o[(8, False)].mean():+.2f}) | "
          f"depth 20: ON {o[(20, True)].mean():.2f} OFF {o[(20, False)].mean():.2f} ({o[(20, True)].mean() - o[(20, False)].mean():+.2f})")
header("Minimum validation loss, constant LR (mean ± SD, n-1)")
for d in (8, 20):
    for s in (True, False):
        v = np.array([min(r["valid_losses"]) for r in runs(V1, d, s)])
        print(f"depth={d:2} residual={'ON ' if s else 'OFF'}: {v.mean():.3f} ± {v.std(ddof=1):.3f}")
