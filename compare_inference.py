import argparse
import json
import torch
import openai
from pathlib import Path

# Maximize determinism and force fp32 accumulation where possible.
# Note: these affect PyTorch ops (linear projections, layernorm, etc.) but NOT
# custom CUDA kernels like FA3 — those manage their own internal precision.
# torch.use_deterministic_algorithms(True, warn_only=True)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

MODEL_PATH = "/scratch/ansh/models/maple_reference_model"

MAX_NEW_TOKENS = 128

BASE_PROMPTS = [
    "What's your name",
    "My name is",
    "I love candy at",
    "Old Macdonald",
    "Mark Zuckerberg",
    "Papa John's Pizza",
    "After KSI left Diddy's party",
    "9 + 10 = "
]


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_hf(model_path: str, prompts: list[str], max_new_tokens: int,
           tokenizer=None, seed: int | None = None) -> tuple[list[str], list[list[int]]]:
    """Returns (decoded_strings, token_id_lists)."""
    from transformers import AutoModelForCausalLM

    if seed is not None:
        torch.manual_seed(seed)

    if tokenizer is None:
        tokenizer = load_tokenizer(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:1",
    )
    model.eval()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    token_ids = [seq[prompt_len:].tolist() for seq in out]
    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
    return decoded, token_ids


def run_sglang(model_path: str, prompts: list[str], max_new_tokens: int, seed: int | None = None) -> list[str]:
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    extra = {"seed": seed} if seed is not None else {}
    response = client.completions.create(
        model=model_path,
        prompt=prompts,
        temperature=0,
        max_tokens=max_new_tokens,
        **extra,
    )
    # choices are returned in order matching the input prompts
    return [choice.text for choice in sorted(response.choices, key=lambda c: c.index)]


def token_match(hf_ids: list[int], sgl_ids: list[int]) -> list[bool]:
    """Per-position match; positions beyond the shorter sequence are False."""
    length = max(len(hf_ids), len(sgl_ids))
    return [
        i < len(hf_ids) and i < len(sgl_ids) and hf_ids[i] == sgl_ids[i]
        for i in range(length)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", type=str, default=None,
                        help="JSON file with a list of prompt strings to append to the base prompts.")
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated indices into the combined prompt list (base + json) to run. E.g. '0,2,5'.")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed passed to torch (HF) and the SGLang API.")
    parser.add_argument("--output", type=str, default="compare_outputs.json",
                        help="Path to write the JSON output file.")
    parser.add_argument("--no_hf", action="store_true", help="Skip the HF run.")
    parser.add_argument("--no_sglang", action="store_true", help="Skip the SGLang run.")
    args = parser.parse_args()

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

    hf_outs, hf_token_ids = None, None
    sglang_outs, sglang_token_ids = None, None

    if not args.no_hf:
        hf_outs, hf_token_ids = run_hf(MODEL_PATH, prompts, args.max_new_tokens,
                                        tokenizer=tokenizer, seed=args.seed)
        print("[HF]")
        for prompt, out in zip(prompts, hf_outs):
            print(f"  {prompt!r} -> {out!r}")

    if not args.no_sglang:
        print("\n[sglang]")
        sglang_outs = run_sglang(MODEL_PATH, prompts, args.max_new_tokens, seed=args.seed)
        sglang_token_ids = [
            tokenizer.encode(text, add_special_tokens=False) for text in sglang_outs
        ]
        for prompt, out in zip(prompts, sglang_outs):
            print(f"  {prompt!r} -> {out!r}")

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
