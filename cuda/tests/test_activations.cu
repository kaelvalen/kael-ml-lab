// ============================================================
// D2L CUDA - Aktivasyon Fonksiyonları Testi
// ============================================================

#include "activations.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    constexpr int N = 1024;
    size_t size = N * sizeof(float);

    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // [-5, 5] arası rastgele değerler
    for (int i = 0; i < N; ++i) {
        h_input[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    printf("=== Aktivasyon Fonksiyonları Testi (N=%d) ===\n", N);

    // ReLU testi
    d2l::cuda::relu_forward(d_input, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    bool relu_ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = fmaxf(0.0f, h_input[i]);
        if (fabsf(h_output[i] - expected) > 1e-5) { relu_ok = false; break; }
    }
    printf("ReLU    : %s\n", relu_ok ? "PASSED ✓" : "FAILED ✗");

    // Sigmoid testi
    d2l::cuda::sigmoid_forward(d_input, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    bool sigmoid_ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = 1.0f / (1.0f + expf(-h_input[i]));
        if (fabsf(h_output[i] - expected) > 1e-5) { sigmoid_ok = false; break; }
    }
    printf("Sigmoid : %s\n", sigmoid_ok ? "PASSED ✓" : "FAILED ✗");

    // Tanh testi
    d2l::cuda::tanh_forward(d_input, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    bool tanh_ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = tanhf(h_input[i]);
        if (fabsf(h_output[i] - expected) > 1e-5) { tanh_ok = false; break; }
    }
    printf("Tanh    : %s\n", tanh_ok ? "PASSED ✓" : "FAILED ✗");

    // Temizlik
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
