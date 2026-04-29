"""
Plot the mean-absolute-diff overview for a specific range of generation steps.

Usage:
    python plot_range.py --start-step 2 --end-step 5
    python plot_range.py --start-step 0 --end-step 0   # single step
    python plot_range.py                                # all steps (same as compare_tensors --plot)

All other args (--hf, --sgl, --rtol, --atol, --diff-plot-out, --tokenizer) mirror
compare_tensors.py so the two scripts can be used interchangeably.
"""
import argparse
import re
import sys
import torch

from compare_tensors import (
    _CHECKPOINT_ORDER_DEF,
    plot_diff,
    plot_diff_per_stage,
)


def _sort_key(k, order):
    parts = k.split("_", 2)
    step  = int(parts[0][1:])
    if len(parts) >= 2 and parts[1].startswith("l") and parts[1][1:].isdigit():
        layer = int(parts[1][1:])
        name  = parts[2] if len(parts) > 2 else ""
    else:
        layer = 999
        name  = "_".join(parts[1:])
    return (step, layer, order.get(name, 99), name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf",  default="/tmp/hf_debug.pt")
    parser.add_argument("--sgl", default="/tmp/sgl_debug.pt")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--start-step", type=int, default=None,
                        help="First generation step to include (inclusive)")
    parser.add_argument("--end-step",   type=int, default=None,
                        help="Last generation step to include (inclusive)")
    parser.add_argument("--diff-plot-out", default="./diff_plot_range.png",
                        help="Output path for the overview plot (default: ./diff_plot_range.png)")
    parser.add_argument("--per-stage", action="store_true",
                        help="Also save individual per-stage plots")
    args = parser.parse_args()

    hf  = torch.load(args.hf,  map_location="cpu", weights_only=True)
    sgl = torch.load(args.sgl, map_location="cpu", weights_only=True)

    _LOGIT_RE = re.compile(r"^s\d{3}_logits$")
    order = _CHECKPOINT_ORDER_DEF

    sk = lambda k: _sort_key(k, order)
    hf_keys  = sorted(hf.keys(),  key=sk)
    sgl_keys = sorted(sgl.keys(), key=sk)

    hf_hidden  = [k for k in hf_keys  if not _LOGIT_RE.match(k)]
    sgl_hidden = [k for k in sgl_keys if not _LOGIT_RE.match(k)]
    common     = sorted(set(hf_hidden) & set(sgl_hidden), key=sk)
    only_hf    = sorted(set(hf_hidden) - set(sgl_hidden), key=sk)
    only_sgl   = sorted(set(sgl_hidden) - set(hf_hidden), key=sk)

    _all_keys = sorted(
        [(k, "common")   for k in common] +
        [(k, "hf_only")  for k in only_hf] +
        [(k, "sgl_only") for k in only_sgl],
        key=lambda x: sk(x[0]),
    )

    def _parse(key):
        parts    = key.split("_", 2)
        step_num = int(parts[0][1:])
        if (len(parts) >= 2 and parts[1].startswith("l")
                and parts[1][1:].isdigit()):
            return step_num, int(parts[1][1:]), parts[2] if len(parts) > 2 else parts[1]
        return step_num, None, "_".join(parts[1:])

    diff_data = []
    for x_pos, (key, source) in enumerate(_all_keys):
        step_num, layer_num, stage = _parse(key)

        if source != "common":
            t = hf[key].float() if source == "hf_only" else sgl[key].float()
            if t.numel() > 0:
                diff_data.append({
                    "x_pos": x_pos, "step": step_num, "layer": layer_num,
                    "stage": stage, "mean_diff": t.abs().mean().item(),
                    "key": key, "source": source,
                })
        else:
            h, s = hf[key].float(), sgl[key].float()
            if h.shape == s.shape:
                diff_data.append({
                    "x_pos": x_pos, "step": step_num, "layer": layer_num,
                    "stage": stage, "mean_diff": (h - s).abs().mean().item(),
                    "key": key, "source": "common",
                })

    # ── step range filter ─────────────────────────────────────────────────────
    start = args.start_step
    end   = args.end_step

    if start is not None or end is not None:
        diff_data = [
            d for d in diff_data
            if (start is None or d["step"] >= start)
            and (end   is None or d["step"] <= end)
        ]
        # Re-index x_pos so the scatter starts from 0
        for new_i, d in enumerate(diff_data):
            d["x_pos"] = new_i

    if not diff_data:
        print("[WARN] No data after step-range filter — nothing to plot.")
        sys.exit(1)

    step_range = sorted(set(d["step"] for d in diff_data))
    print(f"Plotting {len(diff_data)} checkpoints across steps {step_range}")

    plot_diff(diff_data, args.diff_plot_out)
    if args.per_stage:
        plot_diff_per_stage(diff_data, args.diff_plot_out)


if __name__ == "__main__":
    main()
