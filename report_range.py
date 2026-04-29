"""
Print a sequential diff report for a range of generation steps.

Walks every checkpoint in forward-pass order (embed → q/k/v → attn → mlp → …)
for each step in [--start-step, --end-step] and prints per-checkpoint stats.

Usage:
    python report_range.py --start-step 0 --end-step 3
    python report_range.py --start-step 2 --end-step 2  # single step
    python report_range.py                               # all steps

Output columns (common keys):
    [PASS|FAIL|ONLY-HF|ONLY-SGL]  key  mean_abs  max_abs  bad/total  shape

For single-model keys (e.g. attn_probs which is HF-only), the tensor norm
is printed instead of a diff.
"""
import argparse
import re
import sys
import torch

from compare_tensors import _CHECKPOINT_ORDER_DEF


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_LOGIT_RE = re.compile(r"^s\d{3}_logits$")


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


def _parse_key(key):
    parts    = key.split("_", 2)
    step_num = int(parts[0][1:])
    if len(parts) >= 2 and parts[1].startswith("l") and parts[1][1:].isdigit():
        return step_num, int(parts[1][1:]), parts[2] if len(parts) > 2 else parts[1]
    return step_num, None, "_".join(parts[1:])


def _shape_str(t):
    return "×".join(str(s) for s in t.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

_COL_KEY   = 44
_COL_MEAN  = 12
_COL_MAX   = 12
_COL_BAD   = 18
_COL_SHAPE = 20

_HDR = (
    f"  {'status':<10s}  {'key':<{_COL_KEY}s}"
    f"  {'mean_abs':>{_COL_MEAN}s}  {'max_abs':>{_COL_MAX}s}"
    f"  {'bad/total':>{_COL_BAD}s}  shape"
)
_SEP = "  " + "─" * (len(_HDR) - 2)


def _print_header():
    print(_HDR)
    print(_SEP)


def _print_row(marker, key, mean_val, max_val, bad_str, shape_str, *, note=""):
    line = (
        f"  [{marker:<8s}]  {key:<{_COL_KEY}s}"
        f"  {mean_val:>{_COL_MEAN}s}  {max_val:>{_COL_MAX}s}"
        f"  {bad_str:>{_COL_BAD}s}  {shape_str}"
    )
    if note:
        line += f"  ← {note}"
    print(line)


def _fmt(v):
    return f"{v:.4e}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sequential diff report for a step range"
    )
    parser.add_argument("--hf",  default="/tmp/hf_debug.pt",
                        help="HF debug tensor file (default: ./hf_debug.pt)")
    parser.add_argument("--sgl", default="/tmp/sgl_debug.pt",
                        help="SGL debug tensor file (default: ./sgl_debug.pt)")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--start-step", type=int, default=None,
                        help="First generation step to include (inclusive)")
    parser.add_argument("--end-step",   type=int, default=None,
                        help="Last generation step to include (inclusive)")
    parser.add_argument("--show-values", action="store_true",
                        help="Print first 8 elements of each mismatched tensor")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer path/name for decoding logit predictions")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"[WARN] could not load tokenizer: {e}")

    hf  = torch.load(args.hf,  map_location="cpu", weights_only=True)
    sgl = torch.load(args.sgl, map_location="cpu", weights_only=True)

    order = _CHECKPOINT_ORDER_DEF
    sk    = lambda k: _sort_key(k, order)

    hf_keys  = sorted(hf.keys(),  key=sk)
    sgl_keys = sorted(sgl.keys(), key=sk)

    # Separate logit keys from hidden-state keys
    hf_hidden  = [k for k in hf_keys  if not _LOGIT_RE.match(k)]
    sgl_hidden = [k for k in sgl_keys if not _LOGIT_RE.match(k)]
    hf_logit   = {k for k in hf_keys  if _LOGIT_RE.match(k)}
    sgl_logit  = {k for k in sgl_keys if _LOGIT_RE.match(k)}

    common  = set(hf_hidden) & set(sgl_hidden)
    only_hf  = set(hf_hidden) - set(sgl_hidden)
    only_sgl = set(sgl_hidden) - set(hf_hidden)

    all_keys = sorted(
        [(k, "common")   for k in common]  +
        [(k, "hf_only")  for k in only_hf] +
        [(k, "sgl_only") for k in only_sgl],
        key=lambda x: sk(x[0]),
    )

    # Apply step range filter
    def _in_range(key):
        step, _, _ = _parse_key(key)
        if args.start_step is not None and step < args.start_step:
            return False
        if args.end_step is not None and step > args.end_step:
            return False
        return True

    all_keys = [(k, src) for k, src in all_keys if _in_range(k)]

    if not all_keys:
        print("[WARN] No checkpoints found for the requested step range.")
        sys.exit(1)

    steps_present = sorted(set(_parse_key(k)[0] for k, _ in all_keys))
    step_range_str = (f"step {steps_present[0]}" if len(steps_present) == 1
                      else f"steps {steps_present[0]}–{steps_present[-1]}")
    print(f"\nReport: {len(all_keys)} checkpoints across {step_range_str}"
          f"  (rtol={args.rtol}, atol={args.atol})\n")

    if only_hf:
        print(f"[INFO] HF-only keys  ({len(only_hf)}): shown as ONLY-HF with tensor norm")
    if only_sgl:
        print(f"[INFO] SGL-only keys ({len(only_sgl)}): shown as ONLY-SGL with tensor norm")
    print()

    pass_count = fail_count = skip_count = 0
    first_fail = None
    cur_step   = None

    for key, source in all_keys:
        step, layer, stage = _parse_key(key)

        # ── Step header ───────────────────────────────────────────────────────
        if step != cur_step:
            if cur_step is not None:
                print()
            cur_step = step
            print(f"{'━'*70}")
            print(f"  Generation step {step}")
            print(f"{'━'*70}")
            _print_header()

        # ── Single-model key (no diff possible) ───────────────────────────────
        if source != "common":
            t = hf[key].float() if source == "hf_only" else sgl[key].float()
            marker = "ONLY-HF" if source == "hf_only" else "ONLY-SGL"
            norm   = t.abs().mean().item() if t.numel() > 0 else float("nan")
            _print_row(marker, key, _fmt(norm), "—", "— (norm only)", _shape_str(t))
            skip_count += 1
            continue

        # ── Common key ────────────────────────────────────────────────────────
        h = hf[key].float()
        s = sgl[key].float()

        if h.shape != s.shape:
            _print_row("SHAPE!", key, "—", "—",
                       f"hf={tuple(h.shape)} sgl={tuple(s.shape)}", "—")
            fail_count += 1
            if first_fail is None:
                first_fail = key
            continue

        abs_diff  = (h - s).abs()
        mean_diff = abs_diff.mean().item()
        max_diff  = abs_diff.max().item()
        num_bad   = int((abs_diff > args.atol + args.rtol * s.abs()).sum().item())
        total     = h.numel()
        match     = num_bad == 0

        marker = "PASS" if match else "FAIL"
        bad_str = f"{num_bad}/{total}"
        _print_row(marker, key, _fmt(mean_diff), _fmt(max_diff), bad_str, _shape_str(h))

        if match:
            pass_count += 1
        else:
            fail_count += 1
            if first_fail is None:
                first_fail = key

        if args.show_values and not match:
            n = min(8, total)
            h_flat = h.flatten()
            s_flat = s.flatten()
            print(f"           hf [0:{n}] = {h_flat[:n].tolist()}")
            print(f"           sgl[0:{n}] = {s_flat[:n].tolist()}")

    # ── Summary ───────────────────────────────────────────────────────────────
    common_count = pass_count + fail_count
    print(f"\n{'═'*70}")
    print(f"  PASS : {pass_count:4d}   FAIL : {fail_count:4d}   "
          f"SINGLE-MODEL : {skip_count:4d}   TOTAL : {len(all_keys)}")
    if first_fail:
        print(f"  First failure : {first_fail}")
    print(f"{'═'*70}")

    # ── Logit / token summary ─────────────────────────────────────────────────
    logit_keys = sorted(
        [k for k in (hf_logit & sgl_logit) if _in_range(k)]
    )
    if logit_keys:
        print(f"\n{'━'*70}")
        print("  Logit predictions")
        print(f"{'━'*70}")
        for key in logit_keys:
            step, _, _ = _parse_key(key)
            h = hf[key].float()
            s = sgl[key].float()
            T = min(h.shape[0], s.shape[0])
            hf_id  = h[-1].argmax().item()
            sgl_id = s[-1].argmax().item()

            hf_str  = (f"  ({tokenizer.decode([hf_id])!r})"  if tokenizer else "")
            sgl_str = (f"  ({tokenizer.decode([sgl_id])!r})" if tokenizer else "")
            agree   = "✓" if hf_id == sgl_id else "✗"

            from compare_tensors import _kl_per_token
            kl_hs = _kl_per_token(h[:T], s[:T])
            kl_sh = _kl_per_token(s[:T], h[:T])

            print(f"\n  step {step}  [{agree}]  {T} context tokens  vocab={h.shape[-1]}")
            print(f"    HF  → token {hf_id}{hf_str}")
            print(f"    SGL → token {sgl_id}{sgl_str}")
            print(f"    KL(HF‖SGL): mean={kl_hs.mean():.4e}  max={kl_hs.max():.4e}"
                  f"  worst_token={kl_hs.argmax().item()}")
            print(f"    KL(SGL‖HF): mean={kl_sh.mean():.4e}  max={kl_sh.max():.4e}"
                  f"  worst_token={kl_sh.argmax().item()}")
        print(f"\n{'━'*70}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
