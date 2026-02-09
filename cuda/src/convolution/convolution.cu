// ============================================================
// D2L CUDA - 2D Konvolüsyon Kernelleri
// ============================================================

#include "convolution.cuh"

namespace d2l {
namespace cuda {

// ── Naive 2D Konvolüsyon ──────────────────────────────────
__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width,
    int kernel_h, int kernel_w,
    int stride, int padding)
{
    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_h * out_w;

    if (idx >= total) return;

    // Indeks çözümleme
    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int c_out = (idx / (out_w * out_h)) % out_channels;
    int n = idx / (out_w * out_h * out_channels);

    float sum = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    float in_val = input[((n * in_channels + c_in) * height + h_in) * width + w_in];
                    float k_val = kernel[((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw];
                    sum += in_val * k_val;
                }
            }
        }
    }

    output[idx] = sum;
}

void conv2d_naive(const float* input, const float* kernel, float* output,
                  int batch_size, int in_channels, int out_channels,
                  int height, int width,
                  int kernel_h, int kernel_w,
                  int stride, int padding)
{
    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    int total = batch_size * out_channels * out_h * out_w;

    int grid = CEIL_DIV(total, BLOCK_SIZE_1D);
    conv2d_naive_kernel<<<grid, BLOCK_SIZE_1D>>>(
        input, kernel, output,
        batch_size, in_channels, out_channels,
        height, width, kernel_h, kernel_w,
        stride, padding);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace d2l
