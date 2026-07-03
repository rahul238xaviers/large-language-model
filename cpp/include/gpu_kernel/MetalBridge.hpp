#pragma once
#include <cstddef>
#include <vector>

namespace metal_bridge {

/**
 * @brief Initialize Metal device and command queue
 *
 * @note Must be called before any other Metal operations
 */
void initialize();



/**
 * @brief Multiply matrices A [M x K] and B [K x N] to produce C [M x N] on GPU
 *
 * @param a Pointer to raw Matrix A memory on host (size M * K)
 * @param b Pointer to raw Matrix B memory on host (size K * N)
 * @param c Pointer to raw Matrix C memory on host (size M * N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
void gemm_ffn(const float *a, const float *b, float *c, size_t M, size_t N, size_t K);

/**
 * @brief Check if Metal device and FFN pipeline state are initialized and available
 *
 * @return true if available, false otherwise
 */
bool is_available();

// Profiling statistics for GPU vs CPU GEMM execution
extern double accum_gpu_time_ms;
extern double accum_cpu_time_ms;
extern size_t count_gpu_calls;
extern size_t count_cpu_calls;

/**
 * @brief Reset cumulative profiling statistics
 */
void reset_profile_stats();

} // namespace metal_bridge
