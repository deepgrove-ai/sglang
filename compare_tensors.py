"""
Compare HF vs sglang debug tensor dumps.

Usage:
    python compare_tensors.py [--hf /tmp/hf_debug.pt] [--sgl /tmp/sgl_debug.pt]
                              [--rtol 1e-3] [--atol 1e-3]
                              [--plot] [--plot-out /tmp/kl_plot.png]
"""
import argparse
import re
import sys
import torch
import torch.nn.functional as F


# ── KL / entropy helpers ──────────────────────────────────────────────────────

def _kl_per_token(p_logits, q_logits):
    """KL(P ‖ Q) per token. Returns [T]."""
    p = F.softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return F.kl_div(log_q, p, reduction="none").sum(-1)



# ── plotting ──────────────────────────────────────────────────────────────────

def plot_kl(hf, sgl, step_keys, plot_out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot")
        return

    # one scalar KL per step: use the last (or only) token's logits from each key
    steps, kls = [], []
    for key in step_keys:
        step_num = int(key.split("_")[0][1:])   # s003 → 3
        h = hf[key].float()[-1]   # [V]
        s = sgl[key].float()[-1]  # [V]
        kl = _kl_per_token(h.unsqueeze(0), s.unsqueeze(0)).item()
        steps.append(step_num)
        kls.append(kl)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, kls, marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("KL(ref ‖ sgl) (nats)")
    ax.set_title("KL divergence vs reference per generation step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {plot_out}")
    plt.close()


# ── tensor comparison ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf",  default="/tmp/hf_debug.pt")
    parser.add_argument("--sgl", default="/tmp/sgl_debug.pt")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--show-values", action="store_true",
                        help="Print first mismatch values even on pass")
    parser.add_argument("--plot", action="store_true",
                        help="Generate KL-per-token plot for logit keys")
    parser.add_argument("--plot-out", default="/tmp/kl_plot.png",
                        help="Output path for KL plot (default: /tmp/kl_plot.png)")
    args = parser.parse_args()

    hf  = torch.load(args.hf,  map_location="cpu", weights_only=True)
    sgl = torch.load(args.sgl, map_location="cpu", weights_only=True)

    # ── sort key helpers ──────────────────────────────────────────────────────
    _CHECKPOINT_ORDER = {
        "post_input_norm":      0,
        "post_attn":            1,
        "post_post_attn_norm":  2,
        "post_mlp":             3,
        "post_residual":        4,
        "final_norm":           5,
        "logits":               6,
    }
    _LOGIT_RE = re.compile(r"^s\d{3}_logits$")

    def _sort_key(k):
        # Formats:
        #   s{step}_l{layer}_{name}   — per-layer checkpoints
        #   s{step}_logits            — final logit dump
        parts = k.split("_", 2)
        step = int(parts[0][1:])
        if len(parts) >= 2 and parts[1].startswith("l") and parts[1][1:].isdigit():
            layer = int(parts[1][1:])
            name  = parts[2] if len(parts) > 2 else ""
        else:
            layer = 999
            name  = "_".join(parts[1:])
        return (step, layer, _CHECKPOINT_ORDER.get(name, 99), name)

    hf_keys  = sorted(hf.keys(),  key=_sort_key)
    sgl_keys = sorted(sgl.keys(), key=_sort_key)

    # Separate logit keys from hidden-state keys for independent handling
    hf_hidden  = [k for k in hf_keys  if not _LOGIT_RE.match(k)]
    sgl_hidden = [k for k in sgl_keys if not _LOGIT_RE.match(k)]
    hf_logit   = {k for k in hf_keys  if _LOGIT_RE.match(k)}
    sgl_logit  = {k for k in sgl_keys if _LOGIT_RE.match(k)}

    # ── key-set diff ──────────────────────────────────────────────────────────
    only_hf  = set(hf_hidden) - set(sgl_hidden)
    only_sgl = set(sgl_hidden) - set(hf_hidden)
    if only_hf:
        print(f"[WARN] keys only in HF  ({len(only_hf)}): {sorted(only_hf)[:10]}")
    if only_sgl:
        print(f"[WARN] keys only in SGL ({len(only_sgl)}): {sorted(only_sgl)[:10]}")

    common = sorted(set(hf_hidden) & set(sgl_hidden), key=_sort_key)
    print(f"\nComparing {len(common)} common keys  (rtol={args.rtol}, atol={args.atol})\n")

    first_fail = None
    pass_count = 0
    fail_count = 0

    for key in common:
        h = hf[key].float()
        s = sgl[key].float()

        if h.shape != s.shape:
            status = f"SHAPE MISMATCH  hf={tuple(h.shape)}  sgl={tuple(s.shape)}"
            match = False
        else:
            match = bool(torch.allclose(h, s, rtol=args.rtol, atol=args.atol))
            if match:
                status = "OK"
            else:
                abs_diff = (h - s).abs()
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()
                num_bad = (abs_diff > args.atol + args.rtol * s.abs()).sum().item()
                status = (f"MISMATCH  max_abs={max_diff:.4e}  mean_abs={mean_diff:.4e}"
                          f"  bad_elements={num_bad}/{h.numel()}")

        if not match:
            fail_count += 1
            marker = "FAIL"
            if first_fail is None:
                first_fail = key
        else:
            pass_count += 1
            marker = "    "

        print(f"  [{marker}] {key:<40s}  {status}")

        if args.show_values and not match:
            h_flat = h.flatten()
            s_flat = s.flatten()
            n = min(8, h_flat.numel())
            print(f"           hf[0:{n}]  = {h_flat[:n].tolist()}")
            print(f"           sgl[0:{n}] = {s_flat[:n].tolist()}")

    print(f"\n{'='*60}")
    print(f"  PASS: {pass_count}   FAIL: {fail_count}   of {len(common)} keys")
    if first_fail:
        print(f"  First failure: {first_fail}")
    print(f"{'='*60}")

    # ── logit KL summary + plot ───────────────────────────────────────────────
    common_logit = sorted(hf_logit & sgl_logit)
    if not common_logit:
        print("\n[INFO] No logit keys found (s{step}_logits). Models need logit dumps.")
    else:
        print(f"\n{'='*60}")
        print(f"  Logit keys: {common_logit}")
        for key in common_logit:
            h = hf[key].float()
            s = sgl[key].float()
            T = min(h.shape[0], s.shape[0])
            kl_hs = _kl_per_token(h[:T], s[:T])
            kl_sh = _kl_per_token(s[:T], h[:T])
            print(f"\n  [{key}]  {T} tokens  vocab={h.shape[-1]}")
            print(f"    KL(HF‖SGL): mean={kl_hs.mean():.4e}  max={kl_hs.max():.4e}"
                  f"  argmax_token={kl_hs.argmax().item()}")
            print(f"    KL(SGL‖HF): mean={kl_sh.mean():.4e}  max={kl_sh.max():.4e}"
                  f"  argmax_token={kl_sh.argmax().item()}")
        print(f"{'='*60}")

        if args.plot:
            plot_kl(hf, sgl, common_logit, args.plot_out)
        else:
            print("  (pass --plot to generate KL-per-token plot)")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
