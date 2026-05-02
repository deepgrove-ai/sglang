import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True, help="Layer number to filter rows")
parser.add_argument("--csv", default="layer_diffs.csv", help="Path to CSV file")
args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df[df["layer"] == args.layer]

if df.empty:
    all_layers = sorted(pd.read_csv(args.csv)["layer"].unique())
    print(f"No rows found for layer={args.layer}. Available layers: {all_layers}")
    raise SystemExit(1)

df = df.copy()
df["ratio"] = df["mean_diff"] / df["std_diff"]
df = df.sort_values("mean_diff", ascending=False)

print(f"\n{'step':>6}  {'operation':<22}  {'mean_diff':>12}  {'std_diff':>12}  {'mean/std':>10}")
print("-" * 70)
for _, row in df.iterrows():
    print(f"{int(row['step']):>6}  {row['operation']:<22}  {row['mean_diff']:>12.6e}  {row['std_diff']:>12.6e}  {row['ratio']:>10.4f}")
