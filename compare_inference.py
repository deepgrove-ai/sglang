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
MAX_NEW_TOKENS = 128


def run_hf(model_path: str, prompt: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_sglang(model_path: str, prompt: str) -> str:
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    response = client.completions.create(
        model=model_path,
        prompt=prompt,
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )
    return response.choices[0].text


if __name__ == "__main__":
    print(f"Prompt: {PROMPT!r}\n")

    print("[HF]")
    hf_out = run_hf(MODEL_PATH, PROMPT)
    print(hf_out)

    print("\n[sglang]")
    sglang_out = run_sglang(MODEL_PATH, PROMPT)
    print(sglang_out)
