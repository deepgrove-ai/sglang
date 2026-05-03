import torch
import openai

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
PROMPT = "The capital of France is"
PROMPTS = [
    "What's your name",
    "My name is",
    "I love candy"
]
MAX_NEW_TOKENS = 128


def run_hf(model_path: str, prompts: list[str]) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


def run_sglang(model_path: str, prompts: list[str]) -> list[str]:
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    response = client.completions.create(
        model=model_path,
        prompt=prompts,
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )
    # choices are returned in order matching the input prompts
    return [choice.text for choice in sorted(response.choices, key=lambda c: c.index)]


if __name__ == "__main__":
    print(f"Prompts: {PROMPTS}\n")

    print("[HF]")
    hf_outs = run_hf(MODEL_PATH, PROMPTS)
    for prompt, out in zip(PROMPTS, hf_outs):
        print(f"  {prompt!r} -> {out!r}")

    print("\n[sglang]")
    sglang_outs = run_sglang(MODEL_PATH, PROMPTS)
    for prompt, out in zip(PROMPTS, sglang_outs):
        print(f"  {prompt!r} -> {out!r}")
