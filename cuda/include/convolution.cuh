#pragma once
// ============================================================
// D2L CUDA Kernels - Konvolüsyon Operasyonları
// ============================================================

#include "common.cuh"

namespace d2l {
namespace cuda {

// Naive 2D konvolüsyon
void conv2d_naive(const float* input, const float* kernel, float* output,
                  int batch_size, int in_channels, int out_channels,
                  int height, int width,
                  int kernel_h, int kernel_w,
                  int stride, int padding);

// Im2col + GEMM tabanlı konvolüsyon
void conv2d_im2col(const float* input, const float* kernel, float* output,
                   int batch_size, int in_channels, int out_channels,
                   int height, int width,
                   int kernel_h, int kernel_w,
                   int stride, int padding);

} // namespace cuda
} // namespace d2l
