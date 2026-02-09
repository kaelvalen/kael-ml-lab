#pragma once
// ============================================================
// D2L CUDA Kernels - Matris İşlemleri
// ============================================================

#include "common.cuh"

namespace d2l {
namespace cuda {

// Naive matris çarpımı
void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K);

// Tiled (shared memory) matris çarpımı
void matmul_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K, int tile_size = 16);

// cuBLAS matris çarpımı (referans)
void matmul_cublas(const float* A, const float* B, float* C,
                   int M, int N, int K);

} // namespace cuda
} // namespace d2l
