"""
Compare HF vs sglang debug tensor dumps.

Usage:
    python compare_tensors.py [--hf /tmp/hf_debug.pt] [--sgl /tmp/sgl_debug.pt]
                              [--rtol 1e-3] [--atol 1e-3]
                              [--plot] [--plot-out ./kl_plot.png]
                              [--diff-plot-out ./diff_plot.png]
"""
import argparse
import re
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
        print("[WARN] matplotlib not available — skipping KL plot")
        return

    steps, kls = [], []
    for key in step_keys:
        step_num = int(key.split("_")[0][1:])
        h = hf[key].float()[-1]
        s = sgl[key].float()[-1]
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
    print(f"[plot] KL saved → {plot_out}")
    plt.close()


def _diff_scatter(ax, diff_data, color_by="stage"):
    """Shared scatter logic used by both the overview and per-stage plots.

    color_by: "stage" → tab10 by stage name (overview)
              "step"  → tab10 by step number (per-stage)
    """
    from matplotlib.lines import Line2D

    if color_by == "stage":
        keys   = list(dict.fromkeys(d["stage"] for d in diff_data))
        getter = lambda d: d["stage"]
    else:
        keys   = sorted(set(d["step"] for d in diff_data))
        getter = lambda d: d["step"]

    cmap   = plt.cm.get_cmap("tab10", max(len(keys), 1))
    colors = {k: cmap(i) for i, k in enumerate(keys)}

    # Alternating bands to mark step boundaries
    step_changes = [0]
    for i in range(1, len(diff_data)):
        if diff_data[i]["step"] != diff_data[i - 1]["step"]:
            step_changes.append(i)
    step_changes.append(len(diff_data))
    for j, (start, end) in enumerate(zip(step_changes, step_changes[1:])):
        if j % 2 == 1:
            ax.axvspan(start - 0.5, end - 0.5, alpha=0.06, color="gray")

    for d in diff_data:
        c      = colors[getter(d)]
        source = d.get("source", "common")
        if source == "common":
            ax.scatter(d["x_pos"], d["mean_diff"],
                       color=c, alpha=0.75, s=28, zorder=3)
        else:
            # hollow marker: single-model key (hf_only / sgl_only)
            ax.scatter(d["x_pos"], d["mean_diff"],
                       facecolors="none", edgecolors=c, linewidths=0.9,
                       alpha=0.85, s=32, zorder=3)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors[k], markersize=8, label=str(k))
        for k in keys
    ]
    return legend_elems


def plot_diff(diff_data, plot_out):
    """Overview scatter: mean |diff| for every checkpoint, colored by stage."""
    if not diff_data:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping diff plot")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    legend_elems = _diff_scatter(ax, diff_data, color_by="stage")
    from matplotlib.lines import Line2D as _L2D
    hollow = _L2D([0], [0], marker="o", color="gray", markerfacecolor="none",
                  markersize=8, label="single-model (norm)")
    ax.legend(handles=legend_elems + [hollow], fontsize=8, loc="upper left")
    ax.set_xlabel("Forward-pass position  (step → layer → stage)")
    ax.set_ylabel("Mean |hf − sgl|  (hollow = single-model norm)")
    ax.set_title("Mean absolute diff per checkpoint")
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150, bbox_inches="tight")
    print(f"[plot] diff overview saved → {plot_out}")
    plt.close()


def plot_diff_per_stage(diff_data, plot_out_base):
    """One PNG per stage: same scatter style as the overview, colored by step."""
    import os
    if not diff_data:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping per-stage diff plots")
        return

    base, ext = os.path.splitext(plot_out_base)
    ext = ext or ".png"

    stages = list(dict.fromkeys(d["stage"] for d in diff_data))

    for stage in stages:
        stage_pts = [d for d in diff_data if d["stage"] == stage]
        if not stage_pts:
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        legend_elems = _diff_scatter(ax, stage_pts, color_by="step")
        ax.legend(handles=legend_elems, fontsize=8, loc="upper left", title="step")
        ax.set_xlabel("Forward-pass position")
        ax.set_ylabel("Mean |hf − sgl|")
        ax.set_title(f"Mean absolute diff — {stage}")
        ax.set_yscale("symlog", linthresh=1e-7)
        ax.grid(True, alpha=0.3)

        out_path = f"{base}_{stage}{ext}"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot] per-stage diff saved → {out_path}")
        plt.close()


# ── token-step table ──────────────────────────────────────────────────────────

def _safe(text):
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") or "∅"


def _fmt_top5(logits_1d, tokenizer):
    probs = torch.softmax(logits_1d.float(), dim=-1)
    top_probs, top_ids = probs.topk(5)
    parts = []
    for tid, tp in zip(top_ids.tolist(), top_probs.tolist()):
        label = _safe(tokenizer.decode([tid])) if tokenizer else str(tid)
        parts.append(f"{label}({tp * 100:.1f}%)")
    return "  ".join(parts)


def plot_tokens(hf, sgl, step_keys, plot_out, tokenizer=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not available — skipping token plot")
        return

    def _fmt(tok_id):
        if tokenizer:
            return f"{tok_id}  ({_safe(tokenizer.decode([tok_id]))})"
        return str(tok_id)

    rows = []
    for key in sorted(step_keys):
        step_num = int(key.split("_")[0][1:])
        h_logits = hf[key].float()[-1]    # last token → next-token prediction [V]
        s_logits = sgl[key].float()[-1]
        hf_id  = h_logits.argmax().item()
        sgl_id = s_logits.argmax().item()
        rows.append(dict(step=step_num,
                         hf_cell=_fmt(hf_id), sgl_cell=_fmt(sgl_id),
                         hf_top5=_fmt_top5(h_logits, tokenizer),
                         sgl_top5=_fmt_top5(s_logits, tokenizer),
                         match=(hf_id == sgl_id)))

    col_labels = ["Step", "HF token", "SGL token", "=?", "HF top-5", "SGL top-5"]
    table_data = [[str(r["step"]), r["hf_cell"], r["sgl_cell"],
                   "✓" if r["match"] else "✗", r["hf_top5"], r["sgl_top5"]]
                  for r in rows]

    fig_w = 20
    fig_h = max(2.5, 0.35 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    MATCH_COLOR    = "#d4edda"   # pale green
    MISMATCH_COLOR = "#f8d7da"   # pale red
    HEADER_COLOR   = "#dee2e6"

    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor(HEADER_COLOR)
    for row_idx, r in enumerate(rows, start=1):
        color = MATCH_COLOR if r["match"] else MISMATCH_COLOR
        for col_idx in range(len(col_labels)):
            tbl[row_idx, col_idx].set_facecolor(color)

    ax.set_title("Generated token per step — HF vs SGL", fontsize=11, pad=10)
    legend = [mpatches.Patch(color=MATCH_COLOR, label="agree"),
              mpatches.Patch(color=MISMATCH_COLOR, label="differ")]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150, bbox_inches="tight")
    print(f"[plot] token steps saved → {plot_out}")
    plt.close()


# Exported so plot_range.py can import it without re-running main()
# Numbers reflect forward-pass order within a step.
# embed / causal_mask / rope_* are model-level (layer 00) and sort before layer checkpoints.
# q_post_rope / k_post_rope only appear for sliding_attention layers.
_CHECKPOINT_ORDER_DEF = {
    "embed":                  -5,
    "causal_mask":            -4,
    "rope_cos":               -3,
    "rope_sin":               -2,
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
    parser.add_argument("--plot-out", default="./kl_plot.png",
                        help="Output path for KL divergence plot (default: ./kl_plot.png)")
    parser.add_argument("--diff-plot-out", default="./diff_plot.png",
                        help="Output path for mean-diff overview + per-stage plots "
                             "(default: ./diff_plot.png; per-stage saved as diff_plot_{stage}.png)")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer name/path for decoding token IDs to text")
    parser.add_argument("--token-plot-out", default="./token_steps.png",
                        help="Output path for per-step token table (default: ./token_steps.png)")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"[WARN] could not load tokenizer '{args.tokenizer}': {e}")

    hf  = torch.load(args.hf,  map_location="cpu", weights_only=True)
    sgl = torch.load(args.sgl, map_location="cpu", weights_only=True)

    # ── sort key helpers ──────────────────────────────────────────────────────
    _CHECKPOINT_ORDER = _CHECKPOINT_ORDER_DEF
    _LOGIT_RE = re.compile(r"^s\d{3}_output_logits$")

    _EXPERT_RE = re.compile(r"^expert(\d+)_(in|out)$")

    def _sort_key(k):
        # Formats:
        #   s{step}_l{layer}_{name}        — per-layer checkpoints
        #   s{step}_l{layer}_expert{i}_in  — per-expert input
        #   s{step}_l{layer}_expert{i}_out — per-expert output
        #   s{step}_output_logits          — final logit dump
        parts = k.split("_", 2)
        step = int(parts[0][1:])
        if len(parts) >= 2 and parts[1].startswith("l") and parts[1][1:].isdigit():
            layer = int(parts[1][1:])
            name  = parts[2] if len(parts) > 2 else ""
        else:
            layer = 999
            name  = "_".join(parts[1:])
        m = _EXPERT_RE.match(name)
        if m:
            # sort between router_probs (18) and expert_weighted (19) by expert index
            sub = int(m.group(1)) * 2 + (1 if m.group(2) == "in" else 2)
            return (step, layer, 18, sub, name)
        return (step, layer, _CHECKPOINT_ORDER.get(name, 99), 0, name)

    hf_keys  = sorted(hf.keys(),  key=_sort_key)
    sgl_keys = sorted(sgl.keys(), key=_sort_key)

    hf_hidden  = hf_keys
    sgl_hidden = sgl_keys
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

    # Unified sorted sequence: common keys + single-model keys all in forward-pass order.
    # Single-model keys (e.g. attn_probs which is HF-only) are included in the diff plot
    # using their tensor norm as the y-value (hollow markers indicate no true diff).
    _all_keys = sorted(
        [(k, "common")   for k in common] +
        [(k, "hf_only")  for k in only_hf] +
        [(k, "sgl_only") for k in only_sgl],
        key=lambda x: _sort_key(x[0]),
    )

    first_fail = None
    pass_count = 0
    fail_count = 0
    diff_data = []

    for x_pos, (key, source) in enumerate(_all_keys):
        parts    = key.split("_", 2)
        step_num = int(parts[0][1:])
        if (len(parts) >= 2 and parts[1].startswith("l")
                and parts[1][1:].isdigit()):
            layer_num = int(parts[1][1:])
            stage = parts[2] if len(parts) > 2 else parts[1]
        else:
            layer_num = None
            stage = "_".join(parts[1:])

        if source != "common":
            t = hf[key].float() if source == "hf_only" else sgl[key].float()
            if t.numel() > 0:
                diff_data.append({
                    "x_pos": x_pos, "step": step_num, "layer": layer_num,
                    "stage": stage, "mean_diff": t.abs().mean().item(),
                    "key": key, "source": source,
                })
            label = "HF only " if source == "hf_only" else "SGL only"
            print(f"  [{label}] {key:<40s}  norm={t.abs().mean().item():.4e}")
            continue

        h = hf[key].float()
        s = sgl[key].float()

        if h.shape != s.shape:
            status = f"SHAPE MISMATCH  hf={tuple(h.shape)}  sgl={tuple(s.shape)}"
            match = False
        else:
            abs_diff = (h - s).abs()
            mean_diff = abs_diff.mean().item()
            match = bool(torch.allclose(h, s, rtol=args.rtol, atol=args.atol))
            if match:
                status = "OK"
            else:
                max_diff = abs_diff.max().item()
                std_diff = abs_diff.std().item()
                num_bad = (abs_diff > args.atol + args.rtol * s.abs()).sum().item()
                status = (f"MISMATCH  max_abs={max_diff:.4e}  mean_abs={mean_diff:.4e}"
                          f"  std_abs={std_diff:.4e}  bad_elements={num_bad}/{h.numel()}")
            diff_data.append({
                "x_pos": x_pos, "step": step_num, "layer": layer_num,
                "stage": stage, "mean_diff": mean_diff,
                "key": key, "source": "common",
            })

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
        print("\n[INFO] No logit keys found (s{step}_output_logits). Models need logit dumps.")
    else:
        print(f"\n{'='*60}")
        print(f"  Logit keys: {common_logit}")
        for key in common_logit:
            h = hf[key].float()
            s = sgl[key].float()
            T = min(h.shape[0], s.shape[0])
            kl_hs = _kl_per_token(h[:T], s[:T])
            kl_sh = _kl_per_token(s[:T], h[:T])
            hf_pred_id  = h[-1].argmax().item()
            sgl_pred_id = s[-1].argmax().item()
            hf_pred_str  = (f"  ({tokenizer.decode([hf_pred_id])!r})"
                            if tokenizer else "")
            sgl_pred_str = (f"  ({tokenizer.decode([sgl_pred_id])!r})"
                            if tokenizer else "")
            print(f"\n  [{key}]  {T} tokens  vocab={h.shape[-1]}")
            print(f"    HF  predicts token {hf_pred_id}{hf_pred_str}")
            print(f"    SGL predicts token {sgl_pred_id}{sgl_pred_str}")
            print(f"    KL(HF‖SGL): mean={kl_hs.mean():.4e}  max={kl_hs.max():.4e}"
                  f"  argmax_token={kl_hs.argmax().item()}")
            print(f"    KL(SGL‖HF): mean={kl_sh.mean():.4e}  max={kl_sh.max():.4e}"
                  f"  argmax_token={kl_sh.argmax().item()}")
        print(f"{'='*60}")

        if args.plot:
            plot_kl(hf, sgl, common_logit, args.plot_out)
            plot_tokens(hf, sgl, common_logit, args.token_plot_out, tokenizer=tokenizer)
        else:
            print("  (pass --plot to generate KL-per-token plot)")

    if args.plot and diff_data:
        plot_diff(diff_data, args.diff_plot_out)
        plot_diff_per_stage(diff_data, args.diff_plot_out)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
