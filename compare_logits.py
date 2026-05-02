"""Step-by-step logit comparison between HF and SGLang.

Drives the decode loop manually: at each step, HF produces full logits and
SGLang returns top-K logprobs. Both run on the *same* token sequence (greedy
from HF) so the comparison is apples-to-apples.
"""

import torch
import openai

MODEL_PATH = "/scratch/ansh/models/maple_reference_model"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
TOP_LOGPROBS = 20       # how many top-K logprobs to request from SGLang
SGL_BASE_URL = "http://127.0.0.1:30000/v1"


def run_comparison():
    # ── HF setup ─────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading hf tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    client = openai.Client(base_url=SGL_BASE_URL, api_key="None")

    # ── Tokenize prompt ───────────────────────────────────────────────────────
    print("tokenizng prompt")
    enc = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]          # (1, T)
    prompt_text = PROMPT
    past_kv = None

    print(f"Prompt: {PROMPT!r}  ({input_ids.shape[1]} tokens)\n")
    print(f"{'Step':>4}  {'HF token':>20}  {'HF logp':>9}  {'SGL token':>20}  {'SGL logp':>9}  {'match':>5}")
    print("-" * 80)

    for step in range(MAX_NEW_TOKENS):
        # ── HF: one forward pass ─────────────────────────────────────────────
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_kv, use_cache=True)
        logits = out.logits[:, -1, :].float()   # (1, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        next_id = int(logits.argmax(dim=-1))
        hf_logp = float(log_probs[0, next_id])
        hf_tok = tokenizer.decode([next_id])

        # Advance HF state
        past_kv = out.past_key_values
        input_ids = torch.tensor([[next_id]], device=model.device)

        # ── SGLang: one token with top-K logprobs ────────────────────────────
        resp = client.completions.create(
            model=MODEL_PATH,
            prompt=prompt_text,
            max_tokens=1,
            temperature=0,
            logprobs=TOP_LOGPROBS,
        )
        choice = resp.choices[0]
        sgl_tok = choice.text
        sgl_top = choice.logprobs.top_logprobs[0] if choice.logprobs else {}

        # Log-prob SGLang assigned to the *HF-chosen* token (may not be in top-K)
        sgl_logp_for_hf = sgl_top.get(hf_tok, float("nan"))
        sgl_best_tok = max(sgl_top, key=sgl_top.get) if sgl_top else "?"
        sgl_best_logp = sgl_top.get(sgl_best_tok, float("nan"))

        match = "✓" if sgl_tok.strip() == hf_tok.strip() else "✗"
        print(
            f"{step:>4}  {repr(hf_tok):>20}  {hf_logp:>9.4f}"
            f"  {repr(sgl_tok):>20}  {sgl_best_logp:>9.4f}  {match:>5}"
        )
        if sgl_tok.strip() != hf_tok.strip():
            # Show what SGLang's top tokens were
            top5 = sorted(sgl_top.items(), key=lambda x: -x[1])[:5]
            print(f"       SGL top-5: {top5}")
            print(f"       HF logp for SGL-chosen tok {repr(sgl_tok)}: "
                  f"{sgl_top.get(sgl_tok, float('nan')):.4f} vs HF logp {hf_logp:.4f}")

        # Advance SGLang context with the HF-chosen token (keeps sequences in sync)
        prompt_text += hf_tok

        if next_id == tokenizer.eos_token_id:
            print("\n[EOS reached]")
            break


if __name__ == "__main__":
    run_comparison()
