# ============================================================
# D2L CUDA - CMakeLists.txt
# ============================================================

# Include dizini
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── CUDA Kernel Kütüphanesi ────────────────────────────────
add_library(d2l_kernels STATIC
    src/matmul/matmul.cu
    src/activations/activations.cu
    src/convolution/convolution.cu
    src/attention/attention.cu
    src/loss/loss.cu
)

target_include_directories(d2l_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(d2l_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ── Test Executables ───────────────────────────────────────
add_executable(test_matmul tests/test_matmul.cu)
target_link_libraries(test_matmul d2l_kernels)

add_executable(test_activations tests/test_activations.cu)
target_link_libraries(test_activations d2l_kernels)

# ── pybind11 Python Bindings (opsiyonel) ───────────────────
find_package(pybind11 QUIET)
find_package(Torch QUIET)

if(pybind11_FOUND AND Torch_FOUND)
    pybind11_add_module(_cuda_kernels bindings/bindings.cpp)
    target_link_libraries(_cuda_kernels PRIVATE
        d2l_kernels
        ${TORCH_LIBRARIES}
    )
    target_include_directories(_cuda_kernels PRIVATE
        ${TORCH_INCLUDE_DIRS}
    )

    # Python binding'i src/d2l_custom/cuda_ops/ dizinine kopyala
    add_custom_command(TARGET _cuda_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:_cuda_kernels>
        ${CMAKE_SOURCE_DIR}/src/d2l_custom/cuda_ops/
        COMMENT "Copying CUDA bindings to Python package..."
    )
    message(STATUS "pybind11 + PyTorch found → Python bindings will be built")
else()
    message(STATUS "pybind11 or PyTorch not found → Python bindings skipped")
    if(NOT pybind11_FOUND)
        message(STATUS "  → pip install pybind11")
    endif()
    if(NOT Torch_FOUND)
        message(STATUS "  → Set Torch_DIR or install PyTorch")
    endif()
endif()
