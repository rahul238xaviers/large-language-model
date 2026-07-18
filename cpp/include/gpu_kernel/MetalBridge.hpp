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
void gemm_ffn(const float *a, const float *b_gate, const float *b_up, float *c,
              size_t M, size_t N, size_t K);

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
 * @brief Performs RMSNorm forward pass on the GPU.
 */
void rms_norm_forward(const float *input, float *output, const float *weight,
                      float eps, size_t num_rows, size_t dims);

/**
 * @brief Performs RMSNorm backward pass on the GPU.
 */
void rms_norm_backward(const float *grad_output, const float *input,
                       const float *weight, float *grad_input,
                       float *grad_weight, float eps, size_t num_rows,
                       size_t dims);

/**
 * @brief Performs SwiGLU backward pass on the GPU.
 * @param grad_output Gradient of the loss with respect to the output of the
 * SwiGLU layer
 * @param gate Input to the gate activation function
 * @param up Input to the up activation function
 * @param grad_gate Gradient of the loss with respect to the gate activation
 * function
 * @param grad_up Gradient of the loss with respect to the up activation
 * function
 * @param n Number of elements in the input arrays
 */
void swiglu_backward(const float *grad_output, const float *gate,
                     const float *up, float *grad_gate, float *grad_up,
                     size_t n);

/**
 * @brief Performs Rotary Position Embedding (RoPE) backward pass on the GPU in-place.
 */
void rope_backward(float *grad, const float *cos_table, const float *sin_table,
                   size_t batch, size_t heads, size_t seq_len, size_t head_dim);

struct GQABackwardParams {
  uint32_t batch;
  uint32_t n_q_heads;
  uint32_t n_kv_heads;
  uint32_t seq_len;
  uint32_t head_dim;
};

/**
 * @brief Performs GQA attention backward pass on the GPU.
 * Computes gradients for Q, K, and V with atomic accumulation for shared KV heads.
 */
void gqa_backward(const GQABackwardParams &params,
                  const float *Q, const float *K, const float *V,
                  const float *grad_attn_output,
                  float *grad_Q, float *grad_K, float *grad_V);

struct AdamWStepParams {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float bias_correction1;
  float bias_correction2;
  uint32_t n;
};

/**
 * @brief Performs a fused AdamW parameter update step on the GPU.
 * Updates moments, applies weight decay, bias correction, and parameter update.
 */
void adamw_step(float *param, const float *grad, float *m, float *v,
               const AdamWStepParams &params);

/**
 * @brief Start a single grouped command buffer batch for the entire forward
 * step.
 */
void start_batch();

/**
 * @brief Commit the active compute encoder and command buffer, and wait for GPU
 * completion once.
 */
void commit_batch();

} // namespace metal_bridge
