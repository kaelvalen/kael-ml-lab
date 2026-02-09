// ============================================================
// D2L CUDA - Matris Çarpımı Kernelleri
// ============================================================
// Bu dosya, farklı optimizasyon seviyelerinde matris çarpımı
// kernel implementasyonları içerir.
// ============================================================

#include "matmul.cuh"

namespace d2l {
namespace cuda {

// ── Naive Matris Çarpımı ──────────────────────────────────
// Her thread tek bir C[i][j] elemanını hesaplar.
// Zaman karmaşıklığı: O(M*N*K), Global memory bandwidth limited.
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K)
{
    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(CEIL_DIV(N, BLOCK_SIZE_2D), CEIL_DIV(M, BLOCK_SIZE_2D));
    matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}


// ── Tiled Matris Çarpımı (Shared Memory) ──────────────────
// Shared memory kullanarak global memory erişimini azaltır.
// Tile boyutu kadar veriyi shared memory'ye yükler.
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, int TILE)
{
    extern __shared__ float shared[];
    float* As = shared;
    float* Bs = shared + TILE * TILE;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < CEIL_DIV(K, TILE); ++t) {
        // A tile'ını shared memory'ye yükle
        if (row < M && (t * TILE + threadIdx.x) < K)
            As[threadIdx.y * TILE + threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y * TILE + threadIdx.x] = 0.0f;

        // B tile'ını shared memory'ye yükle
        if ((t * TILE + threadIdx.y) < K && col < N)
            Bs[threadIdx.y * TILE + threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y * TILE + threadIdx.x] = 0.0f;

        __syncthreads();

        // Tile içindeki çarpımı hesapla
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y * TILE + k] * Bs[k * TILE + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K, int tile_size)
{
    dim3 block(tile_size, tile_size);
    dim3 grid(CEIL_DIV(N, tile_size), CEIL_DIV(M, tile_size));
    size_t shared_mem = 2 * tile_size * tile_size * sizeof(float);
    matmul_tiled_kernel<<<grid, block, shared_mem>>>(A, B, C, M, N, K, tile_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace d2l
