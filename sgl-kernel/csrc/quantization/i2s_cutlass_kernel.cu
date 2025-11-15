/*
Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <torch/all.h>

#include "sgl_kernel_ops.h"

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cutlass;

// Custom CUDA kernel for I2S unpack + matmul
// This kernel unpacks 2-bit weights on-the-fly during matmul computation
// Weights are packed as: 4 values per byte, each value is 2 bits: {0,1,2} -> {-1,0,1}

template <typename ElementA, typename ElementC, typename ElementAccumulator>
__global__ void i2s_matmul_kernel(
    const ElementA* __restrict__ A,      // Input matrix (M, K)
    const uint8_t* __restrict__ B_packed, // Packed weight matrix (N, num_packed_cols)
    const float* __restrict__ alpha,      // Per-column alpha scales (K,)
    const ElementC* __restrict__ bias,   // Optional bias (N,)
    ElementC* __restrict__ C,            // Output matrix (M, N)
    int M, int N, int K,
    int num_packed_cols,
    int lda, int ldb_packed, int ldc,
    bool has_bias) {
  
  // Tile-based approach similar to CUTLASS
  // Each thread block processes a tile of output
  
  const int TILE_M = 128;
  const int TILE_N = 128;
  const int TILE_K = 32;
  
  const int tid = threadIdx.x;
  const int bid_m = blockIdx.y;
  const int bid_n = blockIdx.x;
  
  const int m_start = bid_m * TILE_M;
  const int n_start = bid_n * TILE_N;
  
  const int m_end = min(m_start + TILE_M, M);
  const int n_end = min(n_start + TILE_N, N);
  
  // Shared memory for tile of A and unpacked B
  __shared__ ElementA tile_A[TILE_M][TILE_K];
  __shared__ float tile_B[TILE_N][TILE_K];
  
  ElementAccumulator acc[TILE_M / 32][TILE_N / 32] = {0};
  
  // Process K dimension in tiles
  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    const int k_end = min(k_tile + TILE_K, K);
    
    // Load tile of A into shared memory
    for (int k_local = tid; k_local < (k_end - k_tile); k_local += blockDim.x) {
      for (int m_local = 0; m_local < (m_end - m_start); ++m_local) {
        if (m_start + m_local < M && k_tile + k_local < K) {
          tile_A[m_local][k_local] = A[(m_start + m_local) * lda + (k_tile + k_local)];
        }
      }
    }
    
    // Load and unpack tile of B into shared memory
    for (int k_local = tid; k_local < (k_end - k_tile); k_local += blockDim.x) {
      const int packed_col_idx = (k_tile + k_local) / 4;
      const int bit_pos = ((k_tile + k_local) % 4) * 2;
      
      if (packed_col_idx < num_packed_cols) {
        for (int n_local = 0; n_local < (n_end - n_start); ++n_local) {
          if (n_start + n_local < N) {
            // Load packed byte
            uint8_t packed_byte = B_packed[(n_start + n_local) * ldb_packed + packed_col_idx];
            
            // Extract 2-bit value
            uint8_t extracted = (packed_byte >> bit_pos) & 0b11;
            
            // Map {0,1,2} -> {-1,0,1}
            float w_val = static_cast<float>(extracted) - 1.0f;
            
            // Apply alpha scaling
            float alpha_val = alpha[k_tile + k_local];
            tile_B[n_local][k_local] = w_val * alpha_val;
          }
        }
      }
    }
    
    __syncthreads();
    
    // Compute matmul for this tile
    for (int k_local = 0; k_local < (k_end - k_tile); ++k_local) {
      for (int m_local = tid / (TILE_N / 32); m_local < (m_end - m_start) && m_local < TILE_M / 32; ++m_local) {
        for (int n_local = tid % (TILE_N / 32); n_local < (n_end - n_start) && n_local < TILE_N / 32; ++n_local) {
          ElementA a_val = tile_A[m_local * 32 + (tid % 32)][k_local];
          float b_val = tile_B[n_local * 32 + (tid / 32)][k_local];
          acc[m_local][n_local] += static_cast<ElementAccumulator>(a_val) * static_cast<ElementAccumulator>(b_val);
        }
      }
    }
    
    __syncthreads();
  }
  
  // Write results
  for (int m_local = tid / (TILE_N / 32); m_local < (m_end - m_start) && m_local < TILE_M / 32; ++m_local) {
    for (int n_local = tid % (TILE_N / 32); n_local < (n_end - n_start) && n_local < TILE_N / 32; ++n_local) {
      int m_idx = m_start + m_local * 32 + (tid % 32);
      int n_idx = n_start + n_local * 32 + (tid / 32);
      
      if (m_idx < M && n_idx < N) {
        ElementAccumulator result = acc[m_local][n_local];
        
        // Add bias if present
        if (has_bias) {
          result += static_cast<ElementAccumulator>(bias[n_idx]);
        }
        
        C[m_idx * ldc + n_idx] = static_cast<ElementC>(result);
      }
    }
  }
}

// Optimized kernel with improved memory access patterns and vectorization
// Each thread block processes a tile of output, with threads cooperating
template <typename ElementA, typename ElementC>
__global__ void i2s_matmul_kernel_v2(
    const ElementA* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const float* __restrict__ alpha,
    const ElementC* __restrict__ bias,
    ElementC* __restrict__ C,
    int M, int N, int K,
    int num_packed_cols,
    int lda, int ldb_packed, int ldc,
    bool has_bias) {
  
  // Tile sizes optimized for memory access and compute
  const int TILE_M = 64;
  const int TILE_N = 64;
  const int TILE_K = 32;
  const int THREADS_PER_BLOCK = 256;  // Use 256 threads, each computes multiple elements
  
  // Each thread block processes a TILE_M x TILE_N output tile
  const int tile_m = blockIdx.y;
  const int tile_n = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;
  
  const int m_start = tile_m * TILE_M;
  const int n_start = tile_n * TILE_N;
  const int m_end = min(m_start + TILE_M, M);
  const int n_end = min(n_start + TILE_N, N);
  
  // Shared memory for tiles
  __shared__ float tile_A[TILE_M][TILE_K + 1];  // +1 for bank conflict avoidance
  __shared__ float tile_B[TILE_N][TILE_K + 1];
  
  // Each thread handles one column (n_local) and computes multiple rows
  // With 256 threads and TILE_N=64, we need 4 threads per column
  const int n_local = tid % TILE_N;
  const int threads_per_col = num_threads / TILE_N;  // Should be 4
  const int col_thread_id = tid / TILE_N;
  const int m_elements_per_thread = (TILE_M + threads_per_col - 1) / threads_per_col;
  const int m_local_start = col_thread_id * m_elements_per_thread;
  
  // Accumulators for multiple output elements per thread (one per row this thread handles)
  float acc[64] = {0.0f};  // Max TILE_M elements
  
  // Process K dimension in tiles
  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    const int k_end = min(k_tile + TILE_K, K);
    const int k_tile_size = k_end - k_tile;
    
    // Cooperative load of A tile into shared memory
    for (int k_local = tid; k_local < k_tile_size; k_local += num_threads) {
      const int k = k_tile + k_local;
      for (int load_m = 0; load_m < (m_end - m_start); ++load_m) {
        const int m_idx = m_start + load_m;
        if (m_idx < M && k < K) {
          tile_A[load_m][k_local] = static_cast<float>(A[m_idx * lda + k]);
        }
      }
    }
    
    // Cooperative load and unpack of B tile into shared memory
    // Process 4 K values at a time (one packed byte)
    for (int packed_idx = tid; packed_idx < (k_tile_size + 3) / 4; packed_idx += num_threads) {
      const int k_base = k_tile + packed_idx * 4;
      const int packed_col_idx = (k_tile + packed_idx * 4) / 4;
      
      if (packed_col_idx < num_packed_cols) {
        for (int n_local = 0; n_local < (n_end - n_start); ++n_local) {
          const int n_idx = n_start + n_local;
          if (n_idx < N) {
            // Load packed byte
            uint8_t packed_byte = B_packed[n_idx * ldb_packed + packed_col_idx];
            
            // Unpack all 4 values from this byte
            for (int i = 0; i < 4; ++i) {
              const int k = k_base + i;
              if (k < k_end && k < K) {
                const int k_local = k - k_tile;
                const int bit_pos = i * 2;
                uint8_t extracted = (packed_byte >> bit_pos) & 0b11;
                float w_val = static_cast<float>(extracted) - 1.0f;
                tile_B[n_local][k_local] = w_val * alpha[k];
              }
            }
          }
        }
      }
    }
    
    __syncthreads();
    
    // Compute partial dot products for this thread's output elements
    if (n_local < (n_end - n_start)) {
      for (int m_local_idx = 0; m_local_idx < m_elements_per_thread; ++m_local_idx) {
        const int m_local = m_local_start + m_local_idx;
        if (m_local < (m_end - m_start) && m_local < TILE_M) {
          const int m = m_start + m_local;
          if (m < M) {
            for (int k_local = 0; k_local < k_tile_size; ++k_local) {
              acc[m_local_idx] += tile_A[m_local][k_local] * tile_B[n_local][k_local];
            }
          }
        }
      }
    }
    
    __syncthreads();
  }
  
  // Write results
  if (n_local < (n_end - n_start)) {
    const int n = n_start + n_local;
    if (n < N) {
      for (int m_local_idx = 0; m_local_idx < m_elements_per_thread; ++m_local_idx) {
        const int m_local = m_local_start + m_local_idx;
        if (m_local < (m_end - m_start) && m_local < TILE_M) {
          const int m = m_start + m_local;
          if (m < M) {
            float result = acc[m_local_idx];
            
            // Add bias if present
            if (has_bias) {
              result += static_cast<float>(bias[n]);
            }
            
            C[m * ldc + n] = static_cast<ElementC>(result);
          }
        }
      }
    }
  }
}

template <typename ElementA, typename ElementC>
void launch_i2s_matmul(
    torch::Tensor const& out,
    torch::Tensor const& x,
    torch::Tensor const& weight_packed,
    torch::Tensor const& alpha,
    torch::Tensor const& bias,
    int64_t K) {
  
  const int M = x.size(0);
  const int N = weight_packed.size(0);
  const int num_packed_cols = weight_packed.size(1);
  
  const ElementA* A_ptr = reinterpret_cast<const ElementA*>(x.data_ptr());
  const uint8_t* B_packed_ptr = reinterpret_cast<const uint8_t*>(weight_packed.data_ptr());
  const float* alpha_ptr = reinterpret_cast<const float*>(alpha.data_ptr());
  const ElementC* bias_ptr = bias.defined() ? reinterpret_cast<const ElementC*>(bias.data_ptr()) : nullptr;
  ElementC* C_ptr = reinterpret_cast<ElementC*>(out.data_ptr());
  
  const int lda = x.stride(0);
  const int ldb_packed = weight_packed.stride(0);
  const int ldc = out.stride(0);
  
  const bool has_bias = bias.defined();
  
  // Launch kernel with tile-based grid
  // Each thread block processes a TILE_M x TILE_N tile
  const int TILE_M = 64;
  const int TILE_N = 64;
  const int threads_per_block = 256;  // 256 threads per block
  const dim3 grid_size((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());
  
  i2s_matmul_kernel_v2<ElementA, ElementC><<<grid_size, threads_per_block, 0, stream>>>(
      A_ptr, B_packed_ptr, alpha_ptr, bias_ptr, C_ptr,
      M, N, K, num_packed_cols,
      lda, ldb_packed, ldc,
      has_bias);
  
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

void i2s_cutlass_matmul(
    torch::Tensor const& out,
    torch::Tensor const& x,
    torch::Tensor const& weight_packed,
    torch::Tensor const& alpha,
    torch::Tensor const& bias,
    int64_t K) {
  
  // Validate inputs
  TORCH_CHECK(x.dim() == 2, "x must be 2D tensor");
  TORCH_CHECK(weight_packed.dim() == 2, "weight_packed must be 2D tensor");
  TORCH_CHECK(alpha.dim() == 1, "alpha must be 1D tensor");
  TORCH_CHECK(alpha.size(0) == K, "alpha size must match K");
  TORCH_CHECK(x.size(1) == K, "x.size(1) must match K");
  TORCH_CHECK(out.size(0) == x.size(0), "out.size(0) must match x.size(0)");
  TORCH_CHECK(out.size(1) == weight_packed.size(0), "out.size(1) must match weight_packed.size(0)");
  
  if (bias.defined()) {
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D tensor");
    TORCH_CHECK(bias.size(0) == weight_packed.size(0), "bias size must match N");
  }
  
  // Dispatch based on data types
  if (x.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16) {
    launch_i2s_matmul<cutlass::half_t, cutlass::half_t>(out, x, weight_packed, alpha, bias, K);
  } else if (x.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16) {
    launch_i2s_matmul<cutlass::bfloat16_t, cutlass::bfloat16_t>(out, x, weight_packed, alpha, bias, K);
  } else if (x.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32) {
    launch_i2s_matmul<float, float>(out, x, weight_packed, alpha, bias, K);
  } else {
    TORCH_CHECK(false, "Unsupported data type combination for I2S matmul");
  }
}
