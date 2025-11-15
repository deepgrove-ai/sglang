from .gguf import (
    ggml_dequantize,
    ggml_moe_a8,
    ggml_moe_a8_vec,
    ggml_moe_get_block_size,
    ggml_mul_mat_a8,
    ggml_mul_mat_vec_a8,
)

try:
    from .i2s import i2s_cutlass_matmul_wrapper
    __all__ = [
        "ggml_dequantize",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
        "ggml_mul_mat_a8",
        "ggml_mul_mat_vec_a8",
        "i2s_cutlass_matmul_wrapper",
    ]
except (ImportError, AttributeError, RuntimeError) as e:
    # i2s module may not be available if C++ kernel wasn't compiled
    __all__ = [
        "ggml_dequantize",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
        "ggml_mul_mat_a8",
        "ggml_mul_mat_vec_a8",
    ]
