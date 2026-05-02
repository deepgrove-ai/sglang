import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, required=True, help="Step value to filter rows")
parser.add_argument("--csv", default="layer_diffs.csv", help="Path to CSV file")
args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df[df["step"] == args.step]

if df.empty:
    print(f"No rows found for step={args.step}. Available steps: {sorted(pd.read_csv(args.csv)['step'].unique())}")
    raise SystemExit(1)

fig, ax = plt.subplots(figsize=(12, 6))

for op, group in df.groupby("operation"):
    group = group.sort_values("layer")
    ax.plot(group["layer"], group["mean_diff"], marker="o", label=op)
    ax.fill_between(
        group["layer"],
        group["mean_diff"] - group["std_diff"],
        group["mean_diff"] + group["std_diff"],
        alpha=0.25,
    )

ax.set_xlabel("Layer")
ax.set_ylabel("Mean Diff")
ax.set_title(f"Layer Diffs — step {args.step}")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"layer_diffs_step{args.step}.png", dpi=150)
print(f"Saved layer_diffs_step{args.step}.png")
plt.show()

ranked = df.copy()
ranked["ratio"] = ranked["mean_diff"] / ranked["std_diff"]
ranked = ranked.sort_values("ratio", ascending=False)
print(f"\n{'layer':>6}  {'operation':<22}  {'mean_diff':>12}  {'std_diff':>12}  {'mean/std':>10}")
print("-" * 70)
for _, row in ranked.iterrows():
    print(f"{int(row['layer']):>6}  {row['operation']:<22}  {row['mean_diff']:>12.6e}  {row['std_diff']:>12.6e}  {row['ratio']:>10.4f}")
