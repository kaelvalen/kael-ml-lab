#pragma once
// ============================================================
// D2L CUDA Kernels - Loss Fonksiyonları
// ============================================================

#include "common.cuh"

namespace d2l {
namespace cuda {

// MSE Loss
void mse_loss_forward(const float* predictions, const float* targets,
                      float* loss, int n);
void mse_loss_backward(const float* predictions, const float* targets,
                       float* grad, int n);

// Cross-Entropy Loss
void cross_entropy_loss_forward(const float* logits, const int* targets,
                                float* loss, int batch_size, int num_classes);
void cross_entropy_loss_backward(const float* softmax_output, const int* targets,
                                 float* grad, int batch_size, int num_classes);

} // namespace cuda
} // namespace d2l
