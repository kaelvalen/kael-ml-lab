// ============================================================
// D2L CUDA - pybind11 Python Bağlantıları
// ============================================================
// CUDA kernellerini Python'dan çağırmak için pybind11 arayüzü.
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "matmul.cuh"
#include "activations.cuh"
#include "convolution.cuh"

namespace py = pybind11;

// ── Matris Çarpımı ────────────────────────────────────────
torch::Tensor matmul_wrapper(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    d2l::cuda::matmul_tiled(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);

    return C;
}

// ── ReLU ──────────────────────────────────────────────────
torch::Tensor relu_wrapper(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    int n = input.numel();

    d2l::cuda::relu_forward(
        input.data_ptr<float>(), output.data_ptr<float>(), n);

    return output;
}

// ── Softmax ───────────────────────────────────────────────
torch::Tensor softmax_wrapper(torch::Tensor input, int dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D for this softmax");

    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int num_classes = input.size(1);

    d2l::cuda::softmax_forward(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, num_classes);

    return output;
}

// ── Module tanımı ─────────────────────────────────────────
PYBIND11_MODULE(_cuda_kernels, m) {
    m.doc() = "D2L CUDA Kernels - Python bindings";

    m.def("matmul", &matmul_wrapper,
          "Tiled matrix multiplication (CUDA)",
          py::arg("A"), py::arg("B"));

    m.def("relu", &relu_wrapper,
          "ReLU activation (CUDA)",
          py::arg("input"));

    m.def("softmax", &softmax_wrapper,
          "Softmax (CUDA)",
          py::arg("input"), py::arg("dim") = -1);
}
