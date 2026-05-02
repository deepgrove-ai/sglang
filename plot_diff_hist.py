import argparse
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--hf",  default="/tmp/hf_debug.pt")
parser.add_argument("--sgl", default="/tmp/sgl_debug.pt")
parser.add_argument("--step",  type=int, required=True)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--op",    required=True)
parser.add_argument("--bins",  type=int, default=100)
parser.add_argument("--out",   default=None)
args = parser.parse_args()

key = f"s{args.step:03d}_l{args.layer:02d}_{args.op}"

hf  = torch.load(args.hf,  map_location="cpu", weights_only=True)
sgl = torch.load(args.sgl, map_location="cpu", weights_only=True)

if key not in hf:
    raise KeyError(f"Key '{key}' not found in {args.hf}. Available (sample): {list(hf.keys())[:10]}")
if key not in sgl:
    raise KeyError(f"Key '{key}' not found in {args.sgl}. Available (sample): {list(sgl.keys())[:10]}")

diff = (hf[key].float() - sgl[key].float()).abs().flatten()
mean = diff.mean().item()
std  = diff.std().item()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(diff.numpy(), bins=args.bins, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(mean, color="red",    linewidth=1.5, label=f"mean = {mean:.4e}")
ax.axvline(mean + std, color="orange", linewidth=1.2, linestyle="--", label=f"mean ± std ({std:.4e})")
ax.axvline(mean - std, color="orange", linewidth=1.2, linestyle="--")
ax.set_xlabel("|hf − sgl|")
ax.set_ylabel("Count")
ax.set_title(f"Abs diff histogram — {key}")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

out = args.out or f"hist_{key}.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"  mean={mean:.6e}  std={std:.6e}  max={diff.max().item():.6e}  n={diff.numel()}")
plt.show()
