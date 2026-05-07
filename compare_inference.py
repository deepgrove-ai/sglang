"""compare_inference.py

Normal mode : compare HF vs SGLang outputs token-by-token, write JSON.
KL mode     : compute per-step KL(HF‖SGLang), print stats, save PNG.
              Enable with --plot-kl.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import openai

MODEL_PATH = "/scratch/ansh/models/maple_reference_model"
MAX_NEW_TOKENS = 1024
TOP_K_LOGPROBS = 1000

BASE_PROMPTS = [
    "What's your name",
    "My name is",
    "I love candy at",
    "Old Macdonald",
    "Mark Zuckerberg",
    "Papa John's Pizza",
    "After KSI left Diddy's party",
    "9 + 10 = "
][:3]


# ── Prompt helpers ────────────────────────────────────────────────────────────

def generate_random_prompts(
    n: int,
    model_path: str,
    seed: int = 42,
    min_tokens: int = 8,
    max_tokens: int = 32,
) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    special_ids = set(tok.all_special_ids)
    valid_ids = [i for i in range(tok.vocab_size) if i not in special_ids]
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        length = rng.randint(min_tokens, max_tokens)
        ids = rng.choices(valid_ids, k=length)
        prompts.append(tok.decode(ids, skip_special_tokens=True))
    return prompts


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ── Model runners ─────────────────────────────────────────────────────────────

def run_hf(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    tokenizer=None,
    seed: int | None = None,
    return_scores: bool = False,
) -> tuple:
    """
    Returns (decoded, token_id_lists, probs_per_seq, lps_per_seq).
    probs_per_seq and lps_per_seq are None when return_scores=False.
    """
    from transformers import AutoModelForCausalLM

    if seed is not None:
        torch.manual_seed(seed)

    if tokenizer is None:
        tokenizer = load_tokenizer(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1",
    )
    model.eval()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=return_scores,
        )

    generated_ids = [seq[prompt_len:].tolist() for seq in out.sequences]
    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    if not return_scores:
        return decoded, generated_ids, None, None

    batch_size = len(prompts)
    all_probs, all_lps = [], []
    for b in range(batch_size):
        probs_b, lps_b = [], []
        for t, score in enumerate(out.scores):
            p = torch.softmax(score[b].float(), dim=-1).cpu()
            lp = torch.log_softmax(score[b].float(), dim=-1).cpu()
            probs_b.append(p)
            lps_b.append(lp[generated_ids[b][t]].item())
        all_probs.append(probs_b)
        all_lps.append(lps_b)

    return decoded, generated_ids, all_probs, all_lps


def run_sglang(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    seed: int | None = None,
    top_k_logprobs: int = 0,
) -> tuple:
    """
    Returns (decoded_strs, sgl_logprobs_or_None).
    sgl_logprobs is a list of (top_logprobs_per_step, token_lps_per_step) when
    top_k_logprobs > 0, otherwise None.
    """
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None", timeout=7200)
    extra: dict = {}
    if seed is not None:
        extra["seed"] = seed
    if top_k_logprobs > 0:
        extra["logprobs"] = top_k_logprobs

    response = client.completions.create(
        model=model_path,
        prompt=prompts,
        temperature=0,
        max_tokens=max_new_tokens,
        **extra,
    )
    choices = sorted(response.choices, key=lambda c: c.index)
    decoded = [c.text for c in choices]

    sgl_logprobs = None
    if top_k_logprobs > 0:
        sgl_logprobs = [
            (c.logprobs.top_logprobs, c.logprobs.token_logprobs)
            for c in choices
        ]

    return decoded, sgl_logprobs


# ── Analysis helpers ──────────────────────────────────────────────────────────

def token_match(hf_ids: list[int], sgl_ids: list[int]) -> list[bool]:
    length = max(len(hf_ids), len(sgl_ids))
    return [
        i < len(hf_ids) and i < len(sgl_ids) and hf_ids[i] == sgl_ids[i]
        for i in range(length)
    ]


def kl_over_topk(p_full: torch.Tensor, top_logprobs: dict, tokenizer) -> float:
    """KL(p ‖ q) over q's top-k support, both sides renormalized."""
    ids, q_lps = [], []
    for tok_str, lp in top_logprobs.items():
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.append(tid)
            q_lps.append(lp)
    if not ids:
        return float("nan")
    ids_t = torch.tensor(ids)
    q = torch.tensor(q_lps).exp().clamp(min=1e-10)
    q = q / q.sum()
    p = p_full[ids_t].clamp(min=1e-10)
    p = p / p.sum()
    return (p * (p / q).log()).sum().item()


def run_kl_analysis(
    prompts: list[str],
    hf_probs,
    hf_step_lps,
    hf_gen_ids,
    sgl_logprobs,
    tokenizer,
    plot_path: str = "kl_plot.png",
):
    import matplotlib.pyplot as plt

    n_seq = len(prompts)
    fig, axes = plt.subplots(n_seq, 2, figsize=(14, 4 * n_seq), squeeze=False)

    for b in range(n_seq):
        sgl_top, sgl_token_lps = sgl_logprobs[b]
        n = min(len(hf_probs[b]), len(sgl_top))
        steps = list(range(n))

        kl_vals = [kl_over_topk(hf_probs[b][i], sgl_top[i], tokenizer) for i in steps]
        lp_diff = [hf_step_lps[b][i] - sgl_token_lps[i] for i in steps]

        valid_kl = [k for k in kl_vals if k == k]
        mean_kl = sum(valid_kl) / len(valid_kl) if valid_kl else float("nan")
        max_kl = max(valid_kl) if valid_kl else float("nan")
        max_kl_step = kl_vals.index(max_kl) if valid_kl else -1

        sgl_top1_ids = [
            tokenizer.convert_tokens_to_ids(max(sgl_top[i], key=sgl_top[i].get))
            for i in steps
        ]
        matches = [hf_gen_ids[b][i] == sgl_top1_ids[i] for i in range(n)]
        match_rate = sum(matches) / len(matches)
        first_mismatch = next((i for i, m in enumerate(matches) if not m), -1)

        stats = (
            f"steps={n}  mean_kl={mean_kl:.6f}  max_kl={max_kl:.6f} @ step {max_kl_step}"
            f"     mean_lp_diff={sum(lp_diff)/len(lp_diff):.6f}"
            f"  max_abs_lp_diff={max(abs(d) for d in lp_diff):.6f}"
            f"     top1_match={match_rate:.1%}  first_mismatch={first_mismatch}"
        )
        print(f"[{b}] {stats}")

        ax1, ax2 = axes[b]
        ax1.plot(steps, kl_vals, marker="o", markersize=2, linewidth=1)
        ax1.set_ylabel("KL(HF ∥ SGLang)")
        ax1.set_title(f"[{b}] {prompts[b][:50]!r}\n{stats}", fontsize=8, loc="left")

        ax2.plot(steps, lp_diff, marker="o", markersize=2, linewidth=1, color="tab:orange")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("log-prob diff (HF − SGLang)")
        ax2.set_xlabel("Decode step")
        ax2.set_title(f"[{b}] log-prob diff")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved → {plot_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Prompt selection
    parser.add_argument("--prompts_json", type=str, default=None,
                        help="JSON file with extra prompt strings appended to base prompts.")
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated indices into the combined prompt list. E.g. '0,2,5'.")
    parser.add_argument("--batch", type=int, default=None,
                        help="Use N random prompts instead of the base list.")
    parser.add_argument("--prompts_file", type=str, default="random_prompts.json",
                        help="JSON file to save/load random prompts (used with --batch).")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=None)

    # Run control
    parser.add_argument("--no_hf", action="store_true", help="Skip the HF run.")
    parser.add_argument("--no_sglang", action="store_true", help="Skip the SGLang run.")

    # Output
    parser.add_argument("--output", type=str, default="compare_outputs.json",
                        help="Path for the token-comparison JSON.")

    # KL mode
    parser.add_argument("--plot-kl", action="store_true", dest="plot_kl",
                        help="Compute per-step KL divergence and save a plot.")
    parser.add_argument("--top_k_logprobs", type=int, default=TOP_K_LOGPROBS,
                        help="Top-k logprobs to request from SGLang (--plot-kl only).")
    parser.add_argument("--kl_output", type=str, default="kl_plot.png",
                        help="Output path for the KL plot PNG.")

    args = parser.parse_args()

    # ── Build prompt list ─────────────────────────────────────────────────────
    if args.batch is not None:
        pf = Path(args.prompts_file)
        if pf.exists():
            saved = json.loads(pf.read_text())
            if len(saved) >= args.batch:
                all_prompts = saved[: args.batch]
                print(f"Loaded {len(all_prompts)} prompts from {pf}")
            else:
                extra = generate_random_prompts(
                    args.batch - len(saved), MODEL_PATH,
                    seed=(args.seed or 42) + len(saved),
                )
                all_prompts = saved + extra
                pf.write_text(json.dumps(all_prompts, indent=2))
                print(f"Extended to {len(all_prompts)} prompts → {pf}")
        else:
            all_prompts = generate_random_prompts(
                args.batch, MODEL_PATH, seed=args.seed or 42
            )
            pf.write_text(json.dumps(all_prompts, indent=2))
            print(f"Generated {len(all_prompts)} prompts → {pf}")
    else:
        all_prompts = list(BASE_PROMPTS)
        if args.prompts_json is not None:
            extra = json.loads(Path(args.prompts_json).read_text())
            print(f"Loaded {len(extra)} prompts from {args.prompts_json}")
            all_prompts.extend(extra)

    if args.include is not None:
        indices = [int(i.strip()) for i in args.include.split(",")]
        prompts = [all_prompts[i] for i in indices]
        print(f"Using {len(prompts)} prompts at indices {indices}")
    else:
        prompts = all_prompts

    print(f"Prompts ({len(prompts)}):")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p!r}")
    print()

    tokenizer = load_tokenizer(MODEL_PATH)

    hf_outs = hf_token_ids = hf_probs = hf_step_lps = None
    sglang_outs = sglang_token_ids = sgl_logprobs = None

    # ── HF run ────────────────────────────────────────────────────────────────
    if not args.no_hf:
        hf_outs, hf_token_ids, hf_probs, hf_step_lps = run_hf(
            MODEL_PATH, prompts, args.max_new_tokens,
            tokenizer=tokenizer, seed=args.seed,
            return_scores=args.plot_kl,
        )
        print("[HF]")
        for prompt, out in zip(prompts, hf_outs):
            print(f"  {prompt!r} -> {out!r}")

    # ── SGLang run ────────────────────────────────────────────────────────────
    if not args.no_sglang:
        print("\n[SGLang]")
        sglang_outs, sgl_logprobs = run_sglang(
            MODEL_PATH, prompts, args.max_new_tokens,
            seed=args.seed,
            top_k_logprobs=args.top_k_logprobs if args.plot_kl else 0,
        )
        sglang_token_ids = [
            tokenizer.encode(text, add_special_tokens=False) for text in sglang_outs
        ]
        for prompt, out in zip(prompts, sglang_outs):
            print(f"  {prompt!r} -> {out!r}")

    # ── Token comparison ──────────────────────────────────────────────────────
    if hf_outs is not None and sglang_outs is not None:
        print("\n\n\n\nResults")
        print("sglang", sglang_outs)
        print("hf    ", hf_outs)
        res = [s == h for s, h in zip(sglang_outs, hf_outs)]
        for i, match in enumerate(res):
            print(f"prompt {i}: {match}")
        print("identical", all(res))

    records = []
    for i, prompt in enumerate(prompts):
        hf_out = hf_outs[i] if hf_outs is not None else None
        sgl_out = sglang_outs[i] if sglang_outs is not None else None
        if hf_token_ids is not None and sglang_token_ids is not None:
            matches = token_match(hf_token_ids[i], sglang_token_ids[i])
        else:
            matches = None
        records.append({
            "prompt": prompt,
            "hf_output": hf_out,
            "sglang_output": sgl_out,
            "tokens_same_vs_hf": matches,
        })

    out_path = Path(args.output)
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nOutputs written to {out_path}")

    # ── KL analysis ───────────────────────────────────────────────────────────
    if args.plot_kl:
        if hf_probs is None or sgl_logprobs is None:
            print("Warning: --plot-kl requires both HF and SGLang runs; skipping KL plot.")
        else:
            run_kl_analysis(
                prompts, hf_probs, hf_step_lps, hf_token_ids,
                sgl_logprobs, tokenizer, plot_path=args.kl_output,
            )
