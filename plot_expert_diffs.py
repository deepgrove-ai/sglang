#!/usr/bin/env python3
"""
Plot per-expert tensor differences from a layer_diff_csv.py output.

For a given step, shows:
  - mean_diff of expert{i}_in  averaged across active experts, per layer
  - mean_diff of expert{i}_out averaged across active experts, per layer
  - mean_diff of expert_weighted, per layer
  - a ranked table of individual experts sorted by mean_diff

Usage:
    python plot_expert_diffs.py --step 60
    python plot_expert_diffs.py --step 60 --csv steps.csv --layer 2
"""
import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd

_EXPERT_RE = re.compile(r"^expert(\d+)_(in|out)$")

parser = argparse.ArgumentParser()
parser.add_argument("--step",  type=int, required=True)
parser.add_argument("--csv",   default="layer_diffs.csv")
parser.add_argument("--layer", type=int, default=None, help="If set, restrict to one layer")
args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df[df["step"] == args.step]
if df.empty:
    print(f"No rows for step={args.step}. Available: {sorted(pd.read_csv(args.csv)['step'].unique())}")
    raise SystemExit(1)

if args.layer is not None:
    df = df[df["layer"] == args.layer]

# Tag each row as in / out / weighted / other
def _tag(op):
    m = _EXPERT_RE.match(op)
    if m:
        return m.group(2), int(m.group(1))   # ("in" or "out", expert_idx)
    if op == "expert_weighted":
        return "weighted", -1
    return None, None

df[["kind", "expert_idx"]] = df["operation"].apply(lambda op: pd.Series(_tag(op)))
expert_df   = df[df["kind"].isin(["in", "out", "weighted"])].copy()

if expert_df.empty:
    print("No expert rows found. Run layer_diff_csv.py first.")
    raise SystemExit(1)

# ── figure 1: mean_diff per layer, averaged across active experts ─────────────
fig, ax = plt.subplots(figsize=(12, 5))

for kind, color in [("in", "steelblue"), ("out", "darkorange"), ("weighted", "green")]:
    sub = expert_df[expert_df["kind"] == kind]
    if sub.empty:
        continue
    agg = sub.groupby("layer")[["mean_diff", "std_diff"]].mean().reset_index()
    agg = agg.sort_values("layer")
    label = f"expert_{kind} (avg over active experts)" if kind != "weighted" else "expert_weighted"
    ax.plot(agg["layer"], agg["mean_diff"], marker="o", color=color, label=label)
    ax.fill_between(
        agg["layer"],
        (agg["mean_diff"] - agg["std_diff"]).clip(lower=0),
        agg["mean_diff"] + agg["std_diff"],
        alpha=0.2, color=color,
    )

ax.set_xlabel("Layer")
ax.set_ylabel("Mean Abs Diff")
title = f"Expert Diffs — step {args.step}"
if args.layer is not None:
    title += f"  layer {args.layer}"
ax.set_title(title)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
out1 = f"expert_diffs_step{args.step}.png" if args.layer is None else f"expert_diffs_step{args.step}_l{args.layer:02d}.png"
plt.savefig(out1, dpi=150)
print(f"Saved {out1}")

# ── figure 2 (optional): per-expert scatter for a single layer ────────────────
if args.layer is not None:
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax2, kind in zip(axes, ["in", "out"]):
        sub = expert_df[(expert_df["kind"] == kind) & (expert_df["layer"] == args.layer)]
        if sub.empty:
            ax2.set_visible(False)
            continue
        sub = sub.sort_values("expert_idx")
        ax2.bar(sub["expert_idx"].astype(str), sub["mean_diff"], color="steelblue" if kind == "in" else "darkorange")
        ax2.set_xlabel("Expert index")
        ax2.set_ylabel("Mean Abs Diff")
        ax2.set_title(f"expert_{kind}  step {args.step}  layer {args.layer}")
        ax2.tick_params(axis="x", labelsize=6, rotation=90)
    plt.tight_layout()
    out2 = f"expert_per_expert_step{args.step}_l{args.layer:02d}.png"
    plt.savefig(out2, dpi=150)
    print(f"Saved {out2}")

# ── ranked table ──────────────────────────────────────────────────────────────
ranked = expert_df[expert_df["kind"].isin(["in", "out"])].copy()
ranked["ratio"] = ranked["mean_diff"] / ranked["std_diff"].replace(0, float("nan"))
ranked = ranked.sort_values("mean_diff", ascending=False)
print(f"\n{'layer':>6}  {'expert':>8}  {'kind':>4}  {'mean_diff':>12}  {'std_diff':>12}  {'mean/std':>10}")
print("-" * 65)
for _, row in ranked.head(30).iterrows():
    print(f"{int(row['layer']):>6}  {int(row['expert_idx']):>8}  {row['kind']:>4}  "
          f"{row['mean_diff']:>12.6e}  {row['std_diff']:>12.6e}  {row['ratio']:>10.4f}")
