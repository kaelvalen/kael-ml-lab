#pragma once
// ============================================================
// D2L CUDA Kernels - Aktivasyon Fonksiyonları
// ============================================================

#include "common.cuh"

namespace d2l {
namespace cuda {

void relu_forward(const float* input, float* output, int n);
void relu_backward(const float* grad_output, const float* input,
                   float* grad_input, int n);

void sigmoid_forward(const float* input, float* output, int n);
void sigmoid_backward(const float* grad_output, const float* output,
                      float* grad_input, int n);

void tanh_forward(const float* input, float* output, int n);
void tanh_backward(const float* grad_output, const float* output,
                   float* grad_input, int n);

void softmax_forward(const float* input, float* output,
                     int batch_size, int num_classes);

} // namespace cuda
} // namespace d2l
