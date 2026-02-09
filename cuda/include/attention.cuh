#pragma once
// ============================================================
// D2L CUDA Kernels - Attention Mekanizması
// ============================================================

#include "common.cuh"

namespace d2l {
namespace cuda {

// Scaled dot-product attention
void scaled_dot_product_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int batch_size, int num_heads,
    int seq_len_q, int seq_len_k,
    int head_dim,
    const float* mask = nullptr);

} // namespace cuda
} // namespace d2l
