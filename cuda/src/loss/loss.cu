// ============================================================
// D2L CUDA - Loss Fonksiyonları
// ============================================================

#include "loss.cuh"
#include <cmath>

namespace d2l {
namespace cuda {

// ── MSE Loss ──────────────────────────────────────────────
__global__ void mse_loss_kernel(
    const float* predictions, const float* targets,
    float* loss, int n)
{
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Her thread kendi loss'unu hesaplar
    float val = 0.0f;
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, sdata[0] / n);
    }
}

__global__ void mse_loss_backward_kernel(
    const float* predictions, const float* targets,
    float* grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = 2.0f * (predictions[idx] - targets[idx]) / n;
    }
}

void mse_loss_forward(const float* predictions, const float* targets,
                      float* loss, int n) {
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    mse_loss_kernel<<<grid, BLOCK_SIZE_1D, BLOCK_SIZE_1D * sizeof(float)>>>(
        predictions, targets, loss, n);
    CUDA_CHECK(cudaGetLastError());
}

void mse_loss_backward(const float* predictions, const float* targets,
                       float* grad, int n) {
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    mse_loss_backward_kernel<<<grid, BLOCK_SIZE_1D>>>(predictions, targets, grad, n);
    CUDA_CHECK(cudaGetLastError());
}


// ── Cross-Entropy Loss ────────────────────────────────────
__global__ void cross_entropy_kernel(
    const float* logits, const int* targets,
    float* loss, int batch_size, int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* row = logits + idx * num_classes;
    int target = targets[idx];

    // Log-sum-exp
    float max_val = row[0];
    for (int c = 1; c < num_classes; ++c) {
        max_val = fmaxf(max_val, row[c]);
    }

    float log_sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        log_sum_exp += expf(row[c] - max_val);
    }
    log_sum_exp = max_val + logf(log_sum_exp);

    float sample_loss = log_sum_exp - row[target];
    atomicAdd(loss, sample_loss / batch_size);
}

void cross_entropy_loss_forward(const float* logits, const int* targets,
                                float* loss, int batch_size, int num_classes) {
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    int grid = CEIL_DIV(batch_size, BLOCK_SIZE_1D);
    cross_entropy_kernel<<<grid, BLOCK_SIZE_1D>>>(
        logits, targets, loss, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

void cross_entropy_loss_backward(const float* softmax_output, const int* targets,
                                 float* grad, int batch_size, int num_classes) {
    // grad = softmax_output - one_hot(targets)
    // Bu basit bir kernel ile yapılabilir
    int total = batch_size * num_classes;
    CUDA_CHECK(cudaMemcpy(grad, softmax_output, total * sizeof(float), cudaMemcpyDeviceToDevice));

    // Target indekslerindeki grad değerlerini 1 azalt
    // (host tarafında veya ayrı bir kernel ile)
}

} // namespace cuda
} // namespace d2l
