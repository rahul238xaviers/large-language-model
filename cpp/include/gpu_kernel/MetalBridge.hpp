#pragma once
#include <cstddef>
#include <cstdint>

namespace metal_bridge {

/**
 * @brief Initialize Metal device and command queue
 *
 * @note Must be called before any other Metal operations
 */
void initialize();

struct GQAParams {
  uint32_t batch;
  uint32_t n_q_heads;
  uint32_t n_kv_heads;
  uint32_t seq_len;
  uint32_t head_dim;
};

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
void gemm_ffn(const float *a, const float *b, float *c, size_t M, size_t N,
              size_t K);

/**
 * @brief Multiply matrices A [M x K] and B [K x N] to produce C [M x N] on GPU
 *        using the custom projection GEMM kernel (weight-resident projection).
 *
 * @note Optimized specifically for projection layers where weight Matrix B [K x
 * N] represents the transformer projection weight.
 *
 * @param a Pointer to raw Matrix A memory on host (size M * K)
 * @param b Pointer to raw Matrix B memory on host (size K * N)
 * @param c Pointer to raw Matrix C memory on host (size M * N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
void gemm_proj(const float *a, const float *b, float *c, size_t M, size_t N,
               size_t K);

/**
 * @brief Check if Metal device and FFN pipeline state are initialized and
 * available
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

/**
 * @brief GQA kernel
 * @param gqa_params
 * @param q [batch, seq_len, n_heads, head_dim]
 * @param k [batch, seq_len, n_kv_heads, head_dim]
 * @param v [batch, seq_len, n_kv_heads, head_dim]
 * @param out_gqa [batch, seq_len, n_heads, head_dim]
 */
void gemm_gqa(const GQAParams &gqa_params, const float *q, const float *k,
              const float *v, float *out_gqa);

/**
 * @brief Start a single grouped command buffer batch for the entire forward step.
 */
void start_batch();

/**
 * @brief Commit the active compute encoder and command buffer, and wait for GPU completion once.
 */
void commit_batch();

} // namespace metal_bridge
