// ============================================================
// D2L CUDA - Attention Mekanizması
// ============================================================

#include "attention.cuh"
#include <cmath>

namespace d2l {
namespace cuda {

// ── Scaled Dot-Product Attention ──────────────────────────
// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
__global__ void attention_scores_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int seq_len_q, int seq_len_k, int head_dim,
    float scale)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len_q && col < seq_len_k) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += Q[row * head_dim + d] * K[col * head_dim + d];
        }
        scores[row * seq_len_k + col] = sum * scale;
    }
}

__global__ void attention_softmax_kernel(
    float* scores,
    const float* mask,
    int seq_len_q, int seq_len_k)
{
    int row = blockIdx.x;
    if (row >= seq_len_q) return;

    float* row_scores = scores + row * seq_len_k;

    // Mask uygula
    if (mask != nullptr) {
        for (int j = 0; j < seq_len_k; ++j) {
            if (mask[row * seq_len_k + j] == 0.0f) {
                row_scores[j] = -1e9f;
            }
        }
    }

    // Max bul
    float max_val = row_scores[0];
    for (int j = 1; j < seq_len_k; ++j) {
        max_val = fmaxf(max_val, row_scores[j]);
    }

    // Exp ve toplam
    float sum = 0.0f;
    for (int j = 0; j < seq_len_k; ++j) {
        row_scores[j] = expf(row_scores[j] - max_val);
        sum += row_scores[j];
    }

    // Normalize
    for (int j = 0; j < seq_len_k; ++j) {
        row_scores[j] /= sum;
    }
}

__global__ void attention_weighted_sum_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len_q, int seq_len_k, int head_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len_q && d < head_dim) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len_k; ++j) {
            sum += scores[row * seq_len_k + j] * V[j * head_dim + d];
        }
        output[row * head_dim + d] = sum;
    }
}

void scaled_dot_product_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int batch_size, int num_heads,
    int seq_len_q, int seq_len_k,
    int head_dim,
    const float* mask)
{
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    int total_heads = batch_size * num_heads;

    for (int h = 0; h < total_heads; ++h) {
        const float* Q_h = Q + h * seq_len_q * head_dim;
        const float* K_h = K + h * seq_len_k * head_dim;
        const float* V_h = V + h * seq_len_k * head_dim;
        float* out_h = output + h * seq_len_q * head_dim;

        // Geçici scores matrisi
        float* scores;
        CUDA_CHECK(cudaMalloc(&scores, seq_len_q * seq_len_k * sizeof(float)));

        // QK^T / sqrt(dk)
        dim3 block1(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
        dim3 grid1(CEIL_DIV(seq_len_k, BLOCK_SIZE_2D), CEIL_DIV(seq_len_q, BLOCK_SIZE_2D));
        attention_scores_kernel<<<grid1, block1>>>(
            Q_h, K_h, scores, seq_len_q, seq_len_k, head_dim, scale);

        // Softmax
        attention_softmax_kernel<<<seq_len_q, 1>>>(scores, mask, seq_len_q, seq_len_k);

        // Weighted sum
        dim3 block2(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
        dim3 grid2(CEIL_DIV(head_dim, BLOCK_SIZE_2D), CEIL_DIV(seq_len_q, BLOCK_SIZE_2D));
        attention_weighted_sum_kernel<<<grid2, block2>>>(
            scores, V_h, out_h, seq_len_q, seq_len_k, head_dim);

        CUDA_CHECK(cudaFree(scores));
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace d2l
