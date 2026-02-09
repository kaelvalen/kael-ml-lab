// ============================================================
// D2L CUDA - Aktivasyon Fonksiyonları
// ============================================================

#include "activations.cuh"
#include <cmath>

namespace d2l {
namespace cuda {

// ── ReLU ──────────────────────────────────────────────────
__global__ void relu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(
    const float* grad_output, const float* input,
    float* grad_input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

void relu_forward(const float* input, float* output, int n) {
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    relu_forward_kernel<<<grid, BLOCK_SIZE_1D>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward(const float* grad_output, const float* input,
                   float* grad_input, int n) {
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    relu_backward_kernel<<<grid, BLOCK_SIZE_1D>>>(grad_output, input, grad_input, n);
    CUDA_CHECK(cudaGetLastError());
}


// ── Sigmoid ───────────────────────────────────────────────
__global__ void sigmoid_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoid_backward_kernel(
    const float* grad_output, const float* output,
    float* grad_input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
    }
}

void sigmoid_forward(const float* input, float* output, int n) {
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    sigmoid_forward_kernel<<<grid, BLOCK_SIZE_1D>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward(const float* grad_output, const float* output,
                      float* grad_input, int n) {
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    sigmoid_backward_kernel<<<grid, BLOCK_SIZE_1D>>>(grad_output, output, grad_input, n);
    CUDA_CHECK(cudaGetLastError());
}


// ── Tanh ──────────────────────────────────────────────────
__global__ void tanh_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanh_backward_kernel(
    const float* grad_output, const float* output,
    float* grad_input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * (1.0f - output[idx] * output[idx]);
    }
}

void tanh_forward(const float* input, float* output, int n) {
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    tanh_forward_kernel<<<grid, BLOCK_SIZE_1D>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void tanh_backward(const float* grad_output, const float* output,
                   float* grad_input, int n) {
    int grid = CEIL_DIV(n, BLOCK_SIZE_1D);
    tanh_backward_kernel<<<grid, BLOCK_SIZE_1D>>>(grad_output, output, grad_input, n);
    CUDA_CHECK(cudaGetLastError());
}


// ── Softmax ───────────────────────────────────────────────
__global__ void softmax_kernel(
    const float* input, float* output,
    int batch_size, int num_classes)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* in_row = input + batch_idx * num_classes;
    float* out_row = output + batch_idx * num_classes;

    // Max bulma (numerik stabilite)
    float max_val = in_row[0];
    for (int i = 1; i < num_classes; ++i) {
        max_val = fmaxf(max_val, in_row[i]);
    }

    // Exp ve toplam
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        out_row[i] = expf(in_row[i] - max_val);
        sum += out_row[i];
    }

    // Normalize
    for (int i = 0; i < num_classes; ++i) {
        out_row[i] /= sum;
    }
}

void softmax_forward(const float* input, float* output,
                     int batch_size, int num_classes) {
    softmax_kernel<<<batch_size, 1>>>(input, output, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace d2l
