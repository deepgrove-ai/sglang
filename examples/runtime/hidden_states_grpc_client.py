"""
Example gRPC client for extracting hidden states from an SGLang teacher model.

Usage:
    1. Start the SGLang gRPC server with hidden states enabled:
       python -m sglang.launch_server \
           --model-path <model> \
           --enable-return-hidden-states \
           --grpc-server \
           --port 30000

    2. Run this client:
       python examples/runtime/hidden_states_grpc_client.py \
           --host localhost --port 30000

For knowledge distillation, this replaces the HTTP /generate endpoint
with a binary gRPC transport that avoids JSON serialization overhead.
"""

import argparse
import time

import grpc
import numpy as np
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc


def hidden_states_proto_to_numpy(hs_proto) -> np.ndarray:
    if hs_proto.HasField("tensor_data"):
        tensor_data = hs_proto.tensor_data
        arr = np.frombuffer(tensor_data.data, dtype=np.dtype(tensor_data.dtype))
        if tensor_data.shape:
            arr = arr.reshape(tuple(tensor_data.shape))
        return arr
    return np.array(hs_proto.values, dtype=np.float32)


def get_hidden_states(
    stub: sglang_scheduler_pb2_grpc.SglangSchedulerStub,
    input_ids: list[int],
    original_text: str = "",
) -> list[np.ndarray]:
    """Send a prefill-only request and return hidden states as numpy arrays."""
    request = sglang_scheduler_pb2.GenerateRequest(
        request_id=f"hs_{time.monotonic_ns()}",
        tokenized=sglang_scheduler_pb2.TokenizedInput(
            input_ids=input_ids,
            original_text=original_text,
        ),
        sampling_params=sglang_scheduler_pb2.SamplingParams(
            max_new_tokens=0,
            temperature=0.0,
        ),
        return_hidden_states=True,
        stream=False,
    )

    hidden_states = []
    for response in stub.Generate(request):
        if response.HasField("error"):
            raise RuntimeError(
                f"Server error: {response.error.message}\n{response.error.details}"
            )
        if response.HasField("complete"):
            for hs_proto in response.complete.all_hidden_states:
                hidden_states.append(hidden_states_proto_to_numpy(hs_proto))

    return hidden_states


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden states via gRPC from SGLang"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        options=[
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ],
    )
    stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(channel)

    # Get model info to determine vocab size
    model_info = stub.GetModelInfo(sglang_scheduler_pb2.GetModelInfoRequest())
    print(f"Model: {model_info.model_path}")
    print(f"Vocab size: {model_info.vocab_size}")

    # Example: prefill-only request with dummy tokens
    input_ids = [1, 2, 3, 4, 5]

    t0 = time.perf_counter()
    hidden_states = get_hidden_states(stub, input_ids, "test prompt")
    elapsed = time.perf_counter() - t0

    print(f"\nReceived {len(hidden_states)} hidden state chunk(s) in {elapsed:.3f}s")
    for i, hs in enumerate(hidden_states):
        print(f"  Chunk {i}: shape={hs.shape}, dtype={hs.dtype}, "
              f"size={hs.nbytes / 1024:.1f} KB")

    channel.close()


if __name__ == "__main__":
    main()
