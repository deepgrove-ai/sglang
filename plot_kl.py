"""Plot KL(HF || SGLang) and generated-token log-prob diff per decode step.

Usage:
    python plot_kl.py                  # single prompt, default config
    python plot_kl.py --batch 4        # generate & save 4 random prompts, run both
    python plot_kl.py --batch 4 --prompts_file my_prompts.json  # reuse saved prompts
"""

import argparse
import json
import random
from pathlib import Path

import openai
import matplotlib.pyplot as plt
import torch

MODEL_PATH = "/scratch/ansh/models/maple_reference_model"
MAX_NEW_TOKENS = 4096
TOP_K_LOGPROBS = 1000


def generate_random_prompts(n: int, model_path: str, seed: int = 42,
                             min_tokens: int = 8, max_tokens: int = 32) -> list[str]:
    """Sample random token sequences from the tokenizer vocab and decode to strings."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Exclude special tokens (BOS, EOS, PAD, UNK and any added_tokens)
    special_ids = set(tok.all_special_ids)
    valid_ids = [i for i in range(tok.vocab_size) if i not in special_ids]

    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        length = rng.randint(min_tokens, max_tokens)
        ids = rng.choices(valid_ids, k=length)
        prompts.append(tok.decode(ids, skip_special_tokens=True))
    return prompts


def get_hf_data(model_path: str, prompts: list[str], max_new_tokens: int):
    """Batched greedy HF decode. Returns per-sequence probs and token log-probs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:2"
    )
    model.eval()

    inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    batch_size = len(prompts)
    generated_ids = out.sequences[:, prompt_len:].tolist()

    all_probs, all_lps, all_gen_ids = [], [], []
    for b in range(batch_size):
        probs_b, lps_b = [], []
        for t, score in enumerate(out.scores):
            p = torch.softmax(score[b].float(), dim=-1).cpu()
            lp = torch.log_softmax(score[b].float(), dim=-1).cpu()
            probs_b.append(p)
            lps_b.append(lp[generated_ids[b][t]].item())
        all_probs.append(probs_b)
        all_lps.append(lps_b)
        all_gen_ids.append(generated_ids[b])

    return all_probs, all_lps, all_gen_ids, tok


def get_sglang_data(model_path: str, prompts: list[str], max_new_tokens: int, top_k: int):
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None", timeout=7200)
    resp = client.completions.create(
        model=model_path,
        prompt=prompts,
        temperature=0,
        max_tokens=max_new_tokens,
        logprobs=top_k,
    )
    # choices are returned in prompt order
    return [(c.logprobs.top_logprobs, c.logprobs.token_logprobs) for c in resp.choices]


def kl_over_topk(p_full: torch.Tensor, top_logprobs: dict, tokenizer) -> float:
    """KL(p || q) over q's top-k support, both sides renormalized."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--prompts_file", type=str, default="random_prompts.json")
    args = parser.parse_args()

    # Load or generate prompts
    pf = Path(args.prompts_file)
    if pf.exists():
        saved = json.loads(pf.read_text())
        if len(saved) >= args.batch:
            prompts = saved[: args.batch]
            print(f"Loaded {len(prompts)} prompts from {pf}")
        else:
            extra = generate_random_prompts(
                args.batch - len(saved), MODEL_PATH, seed=args.seed + len(saved)
            )
            prompts = saved + extra
            pf.write_text(json.dumps(prompts, indent=2))
            print(f"Extended to {len(prompts)} prompts → {pf}")
    else:
        prompts = generate_random_prompts(args.batch, MODEL_PATH, seed=args.seed)
        pf.write_text(json.dumps(prompts, indent=2))
        print(f"Generated {len(prompts)} prompts → {pf}")

    for i, p in enumerate(prompts):
        print(f"  [{i}] {p!r}")


    print("\nQuerying SGLang (batch)...")
    sgl_data = get_sglang_data(MODEL_PATH, prompts, args.max_new_tokens, TOP_K_LOGPROBS)
    for i, (top, _) in enumerate(sgl_data):
        print(f"  [{i}] {len(top)} steps")
        
    print(f"\nRunning HF (batch={len(prompts)})...")
    hf_probs, hf_lps, hf_gen_ids, tokenizer = get_hf_data(MODEL_PATH, prompts, args.max_new_tokens)
    print(f"  {len(hf_probs[0])} steps")

    # ── Summary + plot ────────────────────────────────────────────────────────
    n_seq = len(prompts)
    fig, axes = plt.subplots(n_seq, 2, figsize=(14, 4 * n_seq), squeeze=False)

    print()
    for b in range(n_seq):
        sgl_top, sgl_token_lps = sgl_data[b]
        n = min(len(hf_probs[b]), len(sgl_top))
        steps = list(range(n))

        kl_vals = [kl_over_topk(hf_probs[b][i], sgl_top[i], tokenizer) for i in steps]
        lp_diff = [hf_lps[b][i] - sgl_token_lps[i] for i in steps]

        valid_kl = [k for k in kl_vals if k == k]
        mean_kl = sum(valid_kl) / len(valid_kl) if valid_kl else float("nan")
        max_kl = max(valid_kl) if valid_kl else float("nan")
        max_kl_step = kl_vals.index(max_kl) if valid_kl else -1
        # top-1 match: compare HF greedy token vs SGLang's argmax at each step
        sgl_top1_ids = [
            tokenizer.convert_tokens_to_ids(max(sgl_top[i], key=sgl_top[i].get))
            for i in steps
        ]
        matches = [hf_gen_ids[b][i] == sgl_top1_ids[i] for i in range(n)]
        match_rate = sum(matches) / len(matches)
        first_mismatch = next((i for i, m in enumerate(matches) if not m), -1)

        stats_line = (
            f"steps={n}  mean_kl={mean_kl:.6f}  max_kl={max_kl:.6f} @ step {max_kl_step}"
            f"     mean_lp_diff={sum(lp_diff)/len(lp_diff):.6f}  max_abs_lp_diff={max(abs(d) for d in lp_diff):.6f}"
            f"     top1_match={match_rate:.1%}  first_mismatch={first_mismatch}"
        )
        print(f"[{b}] {stats_line}")

        ax1, ax2 = axes[b]

        ax1.plot(steps, kl_vals, marker="o", markersize=2, linewidth=1)
        ax1.set_ylabel("KL(HF ∥ SGLang)")
        ax1.set_title(f"[{b}] {prompts[b][:50]!r}\n{stats_line}", fontsize=8, loc="left")

        ax2.plot(steps, lp_diff, marker="o", markersize=2, linewidth=1, color="tab:orange")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("log-prob diff (HF − SGLang)")
        ax2.set_xlabel("Decode step")
        ax2.set_title(f"[{b}] log-prob diff")

    plt.tight_layout()
    out_path = "kl_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.show()
