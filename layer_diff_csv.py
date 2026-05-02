#!/usr/bin/env python3
"""
Dump a CSV of per-operation tensor differences for a step range.

Each row is one (step, layer, operation) triple that exists in both debug files.
Rows are ordered by (step, layer, operation position in forward pass).

Usage:
    python layer_diff_csv.py --step-start 60
    python layer_diff_csv.py --step-start 58 --step-end 62 --out steps.csv
"""
import argparse
import csv
import re
import sys
import torch

_CHECKPOINT_ORDER = {
    "layer_in":               -1,
    "post_input_norm":         0,
    "q_proj":                  1,
    "k_proj":                  2,
    "v_proj":                  3,
    "q_post_norm":             4,
    "k_post_norm":             5,
    "q_post_rope":             6,
    "k_post_rope":             7,
    "q":                       8,
    "k":                       9,
    "v":                      10,
    "attn_probs":             11,
    "attn_pre_o":             12,
    "o_proj":                 13,
    "post_attn":              14,
    "post_attn_residual":     15,
    "post_post_attn_norm":    16,
    "router_logits":          17,
    "router_probs":           18,
    # expert{i}_in  (sorted tokens per expert)  — dynamic, sub-sorted within 18 by expert index
    # expert{i}_out (per-expert FFN output)      — dynamic, sub-sorted within 18 by expert index
    "expert_weighted":        19,  # topk weights * expert outputs, before sum
    "ffn_out":                20,
    "post_mlp":               21,
    "post_ffn_residual":      22,
    "layer_out":              23,
    "final_norm":             24,
    "output_logits":          25,
    "output_probs":           26,
}

_LAYER_KEY_RE = re.compile(r"^s(\d{3})_l(\d{2})_(.+)$")
_EXPERT_RE    = re.compile(r"^expert(\d+)_(in|out)$")


def _sort_key(k):
    m = _LAYER_KEY_RE.match(k)
    if m:
        step, layer, name = int(m.group(1)), int(m.group(2)), m.group(3)
        e = _EXPERT_RE.match(name)
        if e:
            sub = int(e.group(1)) * 2 + (1 if e.group(2) == "in" else 2)
            return (step, layer, 18, sub, name)
        return (step, layer, _CHECKPOINT_ORDER.get(name, 99), 0, name)
    return (999, 999, 99, 0, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf",         default="/tmp/hf_debug.pt")
    parser.add_argument("--sgl",        default="/tmp/sgl_debug.pt")
    parser.add_argument("--step-start", type=int, required=True)
    parser.add_argument("--step-end",   type=int, default=None)
    parser.add_argument("--ops",        default=None,
                        help="Comma-separated list of operations to keep, e.g. post_ffn_residual,post_attn_residual")
    parser.add_argument("--out",        default="layer_diffs.csv")
    args = parser.parse_args()

    op_filter = set(args.ops.split(",")) if args.ops else None

    hf  = torch.load(args.hf,  map_location="cpu", weights_only=True)
    sgl = torch.load(args.sgl, map_location="cpu", weights_only=True)

    step_end = args.step_end if args.step_end is not None else args.step_start
    common = sorted(set(hf.keys()) & set(sgl.keys()), key=_sort_key)

    rows = []
    skipped_shape = 0
    for key in common:
        m = _LAYER_KEY_RE.match(key)
        if not m:
            continue
        step  = int(m.group(1))
        layer = int(m.group(2))
        name  = m.group(3)

        if not (args.step_start <= step <= step_end):
            continue
        if op_filter and name not in op_filter:
            continue

        h = hf[key].float()
        s = sgl[key].float()
        if h.shape != s.shape:
            skipped_shape += 1
            continue

        abs_diff = (h - s).abs()
        rows.append({
            "step":      step,
            "layer":     layer,
            "operation": name,
            "mean_diff": f"{abs_diff.mean().item():.6e}",
            "std_diff":  f"{abs_diff.std().item():.6e}",
            "max_diff":  f"{abs_diff.max().item():.6e}",
        })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "layer", "operation", "mean_diff", "std_diff", "max_diff"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}"
          + (f"  ({skipped_shape} skipped: shape mismatch)" if skipped_shape else ""))


if __name__ == "__main__":
    main()
