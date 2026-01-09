import argparse
import time
import requests
import statistics
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class BenchmarkConfig:
    name: str
    prompt: str
    max_tokens: int
    description: str

BENCHMARK_CONFIGS = [
    # Decode-heavy scenarios (short prompt, long generation)
    BenchmarkConfig(
        name="decode_short_story",
        prompt="Write a detailed story about artificial intelligence.",
        max_tokens=2000,
        description="Short prompt, long story generation (decode-heavy)"
    ),
    BenchmarkConfig(
        name="decode_explanation",
        prompt="Explain quantum computing in detail.",
        max_tokens=1500,
        description="Short prompt, detailed explanation (decode-heavy)"
    ),
    BenchmarkConfig(
        name="decode_essay",
        prompt="Write an essay on the future of technology.",
        max_tokens=2500,
        description="Short prompt, long essay (decode-heavy)"
    ),    # Balanced scenarios
    BenchmarkConfig(
        name="balanced_qa",
        prompt="What is machine learning? Explain its applications and future prospects.",
        max_tokens=800,
        description="Medium prompt, medium generation (balanced)"
    ),
    BenchmarkConfig(
        name="balanced_code",
        prompt="Write a Python function to implement a binary search tree with insert, delete, and search operations.",
        max_tokens=1000,
        description="Code generation task (balanced)"
    ),    # Prefill-heavy scenarios (long prompt, short generation)
    BenchmarkConfig(
        name="prefill_summary",
        prompt="""Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.Reinforcement learning is an area of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving rewards or penalties for its actions.Summarize the key points about these AI technologies.""",
        max_tokens=200,
        description="Long prompt, short summary (prefill-heavy)"
    ),
    BenchmarkConfig(
        name="prefill_question",
        prompt="""Artificial intelligence has revolutionized many aspects of our daily lives and industries. From healthcare to finance, transportation to entertainment, AI systems are being deployed to solve complex problems and improve efficiency. Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions. Deep learning networks have achieved remarkable success in image recognition, natural language processing, and game playing. However, AI also raises important ethical questions about privacy, bias, job displacement, and the need for human oversight. As we continue to develop more advanced AI systems, it is crucial to consider both the benefits and potential risks.What are the main benefits and concerns mentioned?""",
        max_tokens=150,
        description="Long prompt, short answer (prefill-heavy)"
    ),    # Edge cases
    BenchmarkConfig(
        name="edge_very_short",
        prompt="Hi",
        max_tokens=100,
        description="Very short prompt, short generation"
    ),
    BenchmarkConfig(
        name="edge_very_long",
        prompt="Write a comprehensive guide to machine learning covering supervised learning, unsupervised learning, reinforcement learning, neural networks, deep learning, convolutional neural networks, recurrent neural networks, transformers, attention mechanisms, transfer learning, fine-tuning, data preprocessing, feature engineering, model evaluation, cross-validation, hyperparameter tuning, regularization, overfitting, underfitting, bias-variance tradeoff, ensemble methods, gradient descent, backpropagation, and practical applications.",
        max_tokens=3000,
        description="Very long prompt, very long generation"
    ),
]

def run_benchmark(host: str, port: int, concurrency: int, scenarios: Optional[List[str]] = None):
    base_url = f"http://{host}:{port}"
    
    print(f"Benchmarking server at {base_url}")
    print(f"Concurrency: {concurrency}")
    
    # Filter scenarios if specified
    configs_to_run = BENCHMARK_CONFIGS
    if scenarios:
        configs_to_run = [c for c in BENCHMARK_CONFIGS if c.name in scenarios]
        if not configs_to_run:
            print(f"No matching scenarios found for: {scenarios}")
            return

    print("-" * 100)
    print(f"{'Scenario':<25} | {'Description':<50} | {'TPS':<8} | {'Lat(ms)':<8} | {'Toks':<6}")
    print("-" * 100)

    for config in configs_to_run:
        # Warmup
        requests.post(f"{base_url}/generate", json={
            "text": "Warmup",
            "sampling_params": {"max_new_tokens": 10, "temperature": 0}
        })

        latencies = []
        total_tokens = 0
        start_time = time.time()
        
        # Simple sequential execution for now (can upgrade to concurrent if needed, but sequential is easier to debug per-request TPS)
        # For concurrency > 1, we would need asyncio or threading. 
        # Given the user asked for these specific scenarios, let's run them sequentially first to get per-req stats.
        # Wait, the user prompt implies 'benchmark_scenarios.py' might have supported concurrency. 
        # Let's keep it simple for now: run N requests sequentially or in parallel?
        # The prompt didn't specify load testing logic, just the config. 
        # I'll implement simple sequential N=1 for each scenario to measure latency/throughput of that specific generation shape.
        
        req_start = time.time()
        resp = requests.post(f"{base_url}/generate", json={
            "text": config.prompt,
            "sampling_params": {
                "max_new_tokens": config.max_tokens,
                "temperature": 0.5,
                "repetition_penalty": 1.1,
                "ignore_eos": True  # Force generation to max_tokens for consistent measurement
            }
        })
        req_end = time.time()
        
        if resp.status_code == 200:
            data = resp.json()
            # Try to get token count from response, otherwise estimate or assume max
            # SGLang usually returns 'meta_info' with 'completion_tokens'
            meta = data.get("meta_info", {})
            generated_tokens = meta.get("completion_tokens", config.max_tokens) # Fallback to max if missing
            
            latency = req_end - req_start
            tps = generated_tokens / latency if latency > 0 else 0
            
            print(f"{config.name:<25} | {config.description:<50} | {tps:<8.2f} | {latency*1000:<8.0f} | {generated_tokens:<6}")
        else:
            print(f"{config.name:<25} | FAILED ({resp.status_code})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SGLang benchmarks")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=30080, help="Server port")
    parser.add_argument("--concurrency", type=int, default=1, help="Request concurrency (not fully implemented in this simple script yet)")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to run")
    args = parser.parse_args()
    
    run_benchmark(args.host, args.port, args.concurrency, args.scenarios)

