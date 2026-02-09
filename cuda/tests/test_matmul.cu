// ============================================================
// D2L CUDA - Matris Çarpımı Testi
// ============================================================

#include "matmul.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

void fill_random(float* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

bool check_result(const float* gpu, const float* cpu, int n, float tol = 1e-3) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) {
            printf("Mismatch at %d: GPU=%.6f CPU=%.6f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int M = 512, N = 512, K = 512;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host bellek
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_naive = (float*)malloc(size_C);
    float* h_C_tiled = (float*)malloc(size_C);
    float* h_C_cpu = (float*)malloc(size_C);

    fill_random(h_A, M * K);
    fill_random(h_B, K * N);

    // CPU referans
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);

    // GPU bellek
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Naive test
    printf("=== Matris Çarpımı Testi (%dx%d x %dx%d) ===\n", M, K, K, N);
    CudaTimer timer;

    timer.tic();
    d2l::cuda::matmul_naive(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    float ms_naive = timer.toc();
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost));
    printf("Naive  : %.3f ms - %s\n", ms_naive,
           check_result(h_C_naive, h_C_cpu, M * N) ? "PASSED ✓" : "FAILED ✗");

    // Tiled test
    timer.tic();
    d2l::cuda::matmul_tiled(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    float ms_tiled = timer.toc();
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost));
    printf("Tiled  : %.3f ms - %s\n", ms_tiled,
           check_result(h_C_tiled, h_C_cpu, M * N) ? "PASSED ✓" : "FAILED ✗");

    printf("Speedup: %.2fx\n", ms_naive / ms_tiled);

    // Temizlik
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_cpu);

    return 0;
}
