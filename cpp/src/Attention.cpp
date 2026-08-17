/**
 * @file Attention.cpp
 * @brief Implementation of the GQA (Grouped Query Attention) layer forward &
 * backward passes
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements the projections, reshaping, GQA forward operations, and
 * analytical backpropagation for Grouped Query Attention (GQA).
 */

#include "Attention.hpp"
#include "RMSNorm.hpp"
#include "Tensor.hpp"
#define ACCELERATE_NEW_LAPACK
#include "gpu_kernel/MetalBridge.hpp"
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <string>
#include <iostream>
#include <dispatch/dispatch.h>
#include <stdexcept>
#include <vector>

/**
 * @brief Construct a new Attention object and initialize the projection
 * weights.
 *
 * Wq_, Wk_, Wv_, and Wo_ are the weight matrices for Query, Key, Value,
 * and Output projections.
 *
 * @param config ModelConfig containing n_heads, n_kv_heads, head_dim, and
 * hidden_dim.
 */
Attention::Attention(const ModelConfig &config)
    : config_(config),
      Wq_({config.hidden_dim, config.n_heads * config.head_dim}, DType::BF16),
      Wk_({config.hidden_dim, config.n_kv_heads * config.head_dim}, DType::BF16),
      Wv_({config.hidden_dim, config.n_kv_heads * config.head_dim}, DType::BF16),
      Wo_({config.n_heads * config.head_dim, config.hidden_dim}, DType::BF16) {

  if (config.n_kv_heads > config.n_heads) {
    throw std::invalid_argument("n_kv_heads cannot be greater than n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::invalid_argument("n_heads must be a multiple of n_kv_heads");
  }
}
/**
 * @brief Reshapes a 3D tensor of shape [batch, seq_len, n_heads * head_dim]
 *        into a 4D tensor of shape [batch, n_heads, seq_len, head_dim].
 *
 * Used to isolate attention heads so self-attention can be computed
 * head-by-head.
 *
 * Example:
 *   If src shape is [2, 128, 256], calling reshape_to_4d(src, 8, 32)
 *   returns a tensor of shape [2, 8, 128, 32].
 *
 * @param src Input 3D tensor to reshape.
 * @param n_heads Number of attention heads.
 * @param head_dim Dimension of each attention head.
 * @return Tensor Reshaped 4D tensor.
 */
Tensor reshape_to_4d(const Tensor &src, size_t n_heads, size_t head_dim) {
  size_t batch = src.shape()[0];
  size_t seq_len = src.shape()[1];

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (use_gpu) {
    Tensor dest({batch, n_heads, seq_len, head_dim}, 0.0f, src.dtype());
    metal_bridge::reshape_to_4d((const float*)src.raw_ptr(), (float*)dest.raw_ptr(),
                                 batch, n_heads, seq_len, head_dim);
    return dest;
  }

  Tensor src_fp32 = src.to_dtype(DType::FP32);
  Tensor dest({batch, n_heads, seq_len, head_dim}, 0.0f, DType::FP32);
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < n_heads; h++) {
      for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          dest(b, h, s, d) = src_fp32(b, s, h * head_dim + d);
        }
      }
    }
  }
  return dest.to_dtype(src.dtype());
}

Tensor reshape_to_3d(const Tensor &src) {
  size_t batch = src.shape()[0];
  size_t n_heads = src.shape()[1];
  size_t seq_len = src.shape()[2];
  size_t head_dim = src.shape()[3];

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (use_gpu) {
    Tensor dest({batch, seq_len, n_heads * head_dim}, 0.0f, src.dtype());
    metal_bridge::reshape_to_3d((const float*)src.raw_ptr(), (float*)dest.raw_ptr(),
                                 batch, n_heads, seq_len, head_dim);
    return dest;
  }

  Tensor src_fp32 = src.to_dtype(DType::FP32);
  Tensor dest({batch, seq_len, n_heads * head_dim}, 0.0f, DType::FP32);
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < n_heads; h++) {
      for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          dest(b, s, h * head_dim + d) = src_fp32(b, h, s, d);
        }
      }
    }
  }
  return dest.to_dtype(src.dtype());
}

/**
 * @brief Performs the forward pass of the Attention layer.
 *
 * Flow:
 * 1. Normalize input using RMSNorm.
 * 2. Project normalized states to Q, K, V.
 * 3. Reshape Q, K, V to 4D to separate heads.
 * 4. Apply Rotary Positional Embeddings (RoPE) to Q and K.
 * 5. Compute GQA scores, apply causal mask, softmax, and weighted sum with V.
 * 6. Concat heads and project with Output projection Wo.
 *
 * @param x Input tensor of shape [batch_size, seq_len, hidden_dim].
 * @param rope Rotary Position Embedding (RoPE) helper.
 * @param cache Optional pointer to Key-Value Cache.
 * @param pos_offset Optional position offset for cache mapping.
 * @return Tensor Attention output of shape [batch_size, seq_len, hidden_dim].
 */
Tensor Attention::forward(const Tensor &x, const RoPE &rope, KVCache *cache,
                          size_t pos_offset) const {

  if (x.shape().size() != 3) {
    throw std::invalid_argument(
        "Input tensor must have 3 dimensions (batch, seq_len, hidden_dim)");
  }
  if (x.shape()[2] != config_.hidden_dim) {
    throw std::invalid_argument(
        "Input tensor hidden dim must match model config");
  }

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  Tensor x_aligned = use_gpu ? x.to_dtype(DType::BF16) : x;
  RMSNorm rms_norm(config_.hidden_dim, config_.rms_norm_eps);
  Tensor x_norm = rms_norm.forward(x_aligned);

  Tensor q4 =
      reshape_to_4d(x_norm.matmul(Wq_), config_.n_heads, config_.head_dim);
  Tensor k4 =
      reshape_to_4d(x_norm.matmul(Wk_), config_.n_kv_heads, config_.head_dim);
  Tensor v4 =
      reshape_to_4d(x_norm.matmul(Wv_), config_.n_kv_heads, config_.head_dim);

  rope.forward(q4, k4);

  size_t batch = x.shape()[0];
  size_t seq_len = x.shape()[1];
  size_t gqa_factor = config_.n_heads / config_.n_kv_heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));

  Tensor attn_output({batch, seq_len, config_.hidden_dim}, 0.0f, x_aligned.dtype());

  const size_t n_h = config_.n_heads;
  const size_t h_d = config_.head_dim;
  const size_t n_kv = config_.n_kv_heads;

  if (use_gpu) {
    auto start = std::chrono::high_resolution_clock::now();
    metal_bridge::flash_attn_fwd((const float*)q4.raw_ptr(), (const float*)k4.raw_ptr(), (const float*)v4.raw_ptr(), (float*)attn_output.raw_ptr(),
                                  batch, n_h, n_kv, seq_len, h_d);
    auto end = std::chrono::high_resolution_clock::now();
    metal_bridge::accum_gpu_time_ms +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
    metal_bridge::count_gpu_calls++;
  } else {
    Tensor q4_fp32 = q4.to_dtype(DType::FP32);
    Tensor k4_fp32 = k4.to_dtype(DType::FP32);
    Tensor v4_fp32 = v4.to_dtype(DType::FP32);
    const float *q4_ptr = q4_fp32.data();
    const float *k4_ptr = k4_fp32.data();
    const float *v4_ptr = v4_fp32.data();
    float *out_ptr = attn_output.data();
    dispatch_apply(
        batch * n_h, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0),
        ^(size_t bh) {
          const size_t b = bh / n_h;
          const size_t h = bh % n_h;
          const size_t kv_h = h / gqa_factor;

          std::vector<float> scores(seq_len, 0.0f); // thread-local scratch

          for (size_t s_q = 0; s_q < seq_len; ++s_q) {
            const float *q_row =
                q4_ptr + (b * n_h * seq_len + h * seq_len + s_q) * h_d;

            float max_score = -INFINITY;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float *k_row =
                  k4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              float dot = 0.0f;
              for (size_t d = 0; d < h_d; ++d)
                dot += q_row[d] * k_row[d];
              scores[s_k] = dot * scale;
              if (scores[s_k] > max_score)
                max_score = scores[s_k];
            }

            // Stable softmax: compute exp in-place, reuse stored values
            float sum_exp = 0.0f;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              scores[s_k] = std::exp(scores[s_k] - max_score);
              sum_exp += scores[s_k];
            }

            // Weighted accumulation — thread-safe: h writes to
            // h*h_d..(h+1)*h_d-1
            float *out_row =
                out_ptr + (b * seq_len + s_q) * (n_h * h_d) + h * h_d;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float prob = scores[s_k] / sum_exp;
              const float *v_row =
                  v4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              for (size_t d = 0; d < h_d; ++d)
                out_row[d] += prob * v_row[d];
            }
          }
        });
  }
  Tensor out = attn_output.matmul(Wo_);
  return use_gpu ? out.to_dtype(x.dtype()) : out;
}
/**
 * @brief Helper function to perform GQA backward pass calculations for a single
 * query token.
 *
 * Implements softmax backward and accumulates gradients w.r.t queries, keys,
 * and values.
 */
static void
gqa_backward_query_token(size_t b, size_t h, size_t kv_h, size_t s_q,
                         size_t seq_len, size_t head_dim, float scale,
                         const Tensor &q4, const Tensor &k4, const Tensor &v4,
                         const Tensor &grad_attn_output, Tensor &grad_q4,
                         Tensor &grad_k4, Tensor &grad_v4) {

  std::vector<float> raw_scores(seq_len);
  std::vector<float> prob(seq_len);
  std::vector<float> dP(seq_len);

  // Step A: Recompute probabilities (prob[s_k])
  float max_score = -INFINITY;
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    float dot = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
      dot += q4(b, h, s_q, d) * k4(b, kv_h, s_k, d);
    }
    raw_scores[s_k] = dot * scale;
    if (raw_scores[s_k] > max_score) {
      max_score = raw_scores[s_k];
    }
  }

  float sum_exp = 0.0f;
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    sum_exp += std::exp(raw_scores[s_k] - max_score);
  }

  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    prob[s_k] = std::exp(raw_scores[s_k] - max_score) / sum_exp;
  }

  // Step B: Backprop w.r.t probabilities (dP) and accumulate grad_v4
  float sum_dP_prob = 0.0f;
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    float dot_g_v = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
      float g = grad_attn_output(b, s_q, h * head_dim + d);
      dot_g_v += g * v4(b, kv_h, s_k, d);
      grad_v4(b, kv_h, s_k, d) += prob[s_k] * g;
    }
    dP[s_k] = dot_g_v;
    sum_dP_prob += dP[s_k] * prob[s_k];
  }

  // Step C: Backprop through Softmax and Scaled Dot-Product to grad_q4 &
  // grad_k4
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    float dS = prob[s_k] * (dP[s_k] - sum_dP_prob);
    float dS_scaled = dS * scale;
    for (size_t d = 0; d < head_dim; ++d) {
      grad_q4(b, h, s_q, d) += dS_scaled * k4(b, kv_h, s_k, d);
      grad_k4(b, kv_h, s_k, d) += dS_scaled * q4(b, h, s_q, d);
    }
  }
}

/**
 * @brief Performs the backward pass of the Attention layer.
 *
 * @param grad_output Gradient from the next layer.
 * @param x Input tensor.
 * @param rope Rotary Position Embedding (RoPE) helper.
 * @param grad_Wq Gradient for Wq.
 * @param grad_Wk Gradient for Wk.
 * @param grad_Wv Gradient for Wv.
 * @param grad_Wo Gradient for Wo.
 * @param cache Optional pointer to Key-Value Cache.
 * @param pos_offset Optional position offset for cache mapping.
 * @return Tensor Gradient for the input tensor x.
 *
 */
Tensor Attention::backward(const Tensor &grad_output, const Tensor &x,
                           const RoPE &rope, Tensor &grad_Wq, Tensor &grad_Wk,
                           Tensor &grad_Wv, Tensor &grad_Wo, KVCache *cache,
                           size_t pos_offset) const {
  if (grad_output.shape() != x.shape()) {
    throw std::invalid_argument("grad_output shape must match input x shape");
  }
  if (grad_Wq.shape() != Wq_.shape()) {
    throw std::invalid_argument("grad_Wq shape must match Wq_ shape");
  }
  if (grad_Wk.shape() != Wk_.shape()) {
    throw std::invalid_argument("grad_Wk shape must match Wk_ shape");
  }
  if (grad_Wv.shape() != Wv_.shape()) {
    throw std::invalid_argument("grad_Wv shape must match Wv_ shape");
  }
  if (grad_Wo.shape() != Wo_.shape()) {
    throw std::invalid_argument("grad_Wo shape must match Wo_ shape");
  }
  size_t batch = x.shape()[0];
  size_t seq_len = x.shape()[1];
  size_t hidden_dim = x.shape()[2];
  size_t n_heads = config_.n_heads;
  size_t n_kv_heads = config_.n_kv_heads;
  size_t head_dim = config_.head_dim;
  size_t gqa_factor = n_heads / n_kv_heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // 1. DType alignment & conversion for GPU path
  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  Tensor x_aligned = use_gpu ? x.to_dtype(DType::BF16) : x;
  Tensor grad_output_aligned = use_gpu ? grad_output.to_dtype(DType::BF16) : grad_output;

  Tensor grad_Wq_bf16 = use_gpu ? grad_Wq.to_dtype(DType::BF16) : grad_Wq;
  Tensor grad_Wk_bf16 = use_gpu ? grad_Wk.to_dtype(DType::BF16) : grad_Wk;
  Tensor grad_Wv_bf16 = use_gpu ? grad_Wv.to_dtype(DType::BF16) : grad_Wv;
  Tensor grad_Wo_bf16 = use_gpu ? grad_Wo.to_dtype(DType::BF16) : grad_Wo;

  RMSNorm rms_norm(hidden_dim, config_.rms_norm_eps);
  Tensor x_norm = rms_norm.forward(x_aligned);

  Tensor q4 = reshape_to_4d(x_norm.matmul(Wq_), n_heads, head_dim);
  Tensor k4 = reshape_to_4d(x_norm.matmul(Wk_), n_kv_heads, head_dim);
  Tensor v4 = reshape_to_4d(x_norm.matmul(Wv_), n_kv_heads, head_dim);
  rope.forward(q4, k4);

  Tensor attn_output({batch, seq_len, hidden_dim}, 0.0f, x_aligned.dtype());

  if (use_gpu) {
    metal_bridge::GQAParams gqa_params = {
        .batch = static_cast<uint32_t>(batch),
        .n_q_heads = static_cast<uint32_t>(n_heads),
        .n_kv_heads = static_cast<uint32_t>(n_kv_heads),
        .seq_len = static_cast<uint32_t>(seq_len),
        .head_dim = static_cast<uint32_t>(head_dim),
    };
    metal_bridge::gemm_gqa(gqa_params, (const float*)q4.raw_ptr(), (const float*)k4.raw_ptr(), (const float*)v4.raw_ptr(), (float*)attn_output.raw_ptr());
  } else {
    const float *q4_ptr = q4.data();
    const float *k4_ptr = k4.data();
    const float *v4_ptr = v4.data();
    float *out_ptr = attn_output.data();
    const size_t n_h = n_heads;
    const size_t h_d = head_dim;
    const size_t n_kv = n_kv_heads;

    dispatch_apply(
        batch * n_h, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0),
        ^(size_t bh) {
          const size_t b = bh / n_h;
          const size_t h = bh % n_h;
          const size_t kv_h = h / gqa_factor;

          std::vector<float> scores(seq_len, 0.0f);

          for (size_t s_q = 0; s_q < seq_len; ++s_q) {
            const float *q_row =
                q4_ptr + (b * n_h * seq_len + h * seq_len + s_q) * h_d;
            float max_score = -INFINITY;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float *k_row =
                  k4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              float dot = 0.0f;
              for (size_t d = 0; d < h_d; ++d)
                dot += q_row[d] * k_row[d];
              scores[s_k] = dot * scale;
              if (scores[s_k] > max_score)
                max_score = scores[s_k];
            }
            float sum_exp = 0.0f;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              scores[s_k] = std::exp(scores[s_k] - max_score);
              sum_exp += scores[s_k];
            }
            float *out_row =
                out_ptr + (b * seq_len + s_q) * (n_h * h_d) + h * h_d;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float prob = scores[s_k] / sum_exp;
              const float *v_row =
                  v4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              for (size_t d = 0; d < h_d; ++d)
                out_row[d] += prob * v_row[d];
            }
          }
        });
  }

  // 2. grad_Wo[NH*HD, H] = attn_output[B*S, NH*HD].T @ grad_output[B*S, H]
  if (use_gpu) {
    metal_bridge::gemm_backward(
        (const float*)attn_output.raw_ptr(),
        (const float*)grad_output_aligned.raw_ptr(),
        (float*)grad_Wo_bf16.raw_ptr(),
        hidden_dim,
        hidden_dim,
        batch * seq_len
    );
  } else {
    grad_Wo = attn_output.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_output_aligned.reshape({batch * seq_len, hidden_dim}));
  }

  // 3. Backpropagate to attention head outputs: grad_attn_output = grad_output @ Wo^T
  Tensor grad_attn_output({batch, seq_len, hidden_dim}, 0.0f, x_aligned.dtype());
  if (use_gpu) {
    metal_bridge::gemm_proj_trans_b(
        (const float*)grad_output_aligned.raw_ptr(),
        (const float*)Wo_.raw_ptr(),
        (float*)grad_attn_output.raw_ptr(),
        batch * seq_len,
        hidden_dim,
        hidden_dim
    );
  } else {
    grad_attn_output = grad_output_aligned.matmul(Wo_.transpose());
  }

  // 4. Initialize head gradients — FP32 because fused_attn_bwd uses 4-byte
  // atomics on dK/dV (bfloat buffers would overflow adjacent memory).
  Tensor grad_q4({batch, n_heads, seq_len, head_dim}, 0.0f, DType::FP32);
  Tensor grad_k4({batch, n_kv_heads, seq_len, head_dim}, 0.0f, DType::FP32);
  Tensor grad_v4({batch, n_kv_heads, seq_len, head_dim}, 0.0f, DType::FP32);

  // 5. GQA attention backpropagation — fused GPU kernel (no global S/P/dS)
  // Single kernel call: computes dQ, dK, dV using tiled shared memory.
  // Eliminates the 2 GB intermediate score/probability/dS buffers entirely.
  if (use_gpu) {
    struct ProfileBlock {
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
        ProfileBlock(std::string name) : name(name), start(std::chrono::high_resolution_clock::now()) {}
        ~ProfileBlock() {
            if (getenv("PROFILE_BWD")) {
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                std::cout << "[PROFILE-BWD]   " << name << " took " << ms << " ms" << std::endl;
            }
        }
    };

    if (getenv("DEBUG_SKIP_FUSED_BWD")) {
      // CPU GQA backward fallback — used to isolate fused_attn_bwd corruption
      Tensor q4_fp32 = q4.to_dtype(DType::FP32);
      Tensor k4_fp32 = k4.to_dtype(DType::FP32);
      Tensor v4_fp32 = v4.to_dtype(DType::FP32);
      for (size_t bb = 0; bb < batch; ++bb) {
        for (size_t hh = 0; hh < n_heads; ++hh) {
          size_t kv_hh = hh / gqa_factor;
          for (size_t s_q = 0; s_q < seq_len; ++s_q) {
            gqa_backward_query_token(bb, hh, kv_hh, s_q, seq_len, head_dim, scale, q4_fp32,
                                     k4_fp32, v4_fp32, grad_attn_output, grad_q4, grad_k4,
                                     grad_v4);
          }
        }
      }
    } else {
      ProfileBlock p("fused_attn_bwd");
      metal_bridge::fused_attn_bwd((const float*)q4.raw_ptr(), (const float*)k4.raw_ptr(), (const float*)v4.raw_ptr(),
                                   (const float*)grad_attn_output.raw_ptr(),
                                   (float*)grad_q4.raw_ptr(), (float*)grad_k4.raw_ptr(), (float*)grad_v4.raw_ptr(),
                                   batch, n_heads, n_kv_heads, seq_len, head_dim);
    }
  } else {
    Tensor q4_fp32 = q4.to_dtype(DType::FP32);
    Tensor k4_fp32 = k4.to_dtype(DType::FP32);
    Tensor v4_fp32 = v4.to_dtype(DType::FP32);
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < n_heads; ++h) {
        size_t kv_h = h / gqa_factor;
        for (size_t s_q = 0; s_q < seq_len; ++s_q) {
          gqa_backward_query_token(b, h, kv_h, s_q, seq_len, head_dim, scale, q4_fp32,
                                   k4_fp32, v4_fp32, grad_attn_output, grad_q4, grad_k4,
                                   grad_v4);
        }
      }
    }
  }

  // 6. Backpropagate RoPE, 7. Reshape gradients back to 3D, and handle BF16 conversions on GPU
  Tensor grad_q_proj_bf16, grad_k_proj_bf16, grad_v_proj_bf16;
  Tensor grad_q_proj, grad_k_proj, grad_v_proj;

  if (use_gpu) {
    Tensor grad_q4_bf16({batch, n_heads, seq_len, head_dim}, 0.0f, DType::BF16);
    Tensor grad_k4_bf16({batch, n_kv_heads, seq_len, head_dim}, 0.0f, DType::BF16);
    Tensor grad_v4_bf16({batch, n_kv_heads, seq_len, head_dim}, 0.0f, DType::BF16);

    metal_bridge::convert_fp32_to_bf16((const float*)grad_q4.raw_ptr(), (float*)grad_q4_bf16.raw_ptr(), grad_q4.size());
    metal_bridge::convert_fp32_to_bf16((const float*)grad_k4.raw_ptr(), (float*)grad_k4_bf16.raw_ptr(), grad_k4.size());
    metal_bridge::convert_fp32_to_bf16((const float*)grad_v4.raw_ptr(), (float*)grad_v4_bf16.raw_ptr(), grad_v4.size());

    rope.backward(grad_q4_bf16, grad_k4_bf16);

    grad_q_proj_bf16 = reshape_to_3d(grad_q4_bf16);
    grad_k_proj_bf16 = reshape_to_3d(grad_k4_bf16);
    grad_v_proj_bf16 = reshape_to_3d(grad_v4_bf16);
  } else {
    rope.backward(grad_q4, grad_k4);
    grad_q_proj = reshape_to_3d(grad_q4);
    grad_k_proj = reshape_to_3d(grad_k4);
    grad_v_proj = reshape_to_3d(grad_v4);
  }

  // 8. Parameter gradients for Wq, Wk, Wv
  if (use_gpu) {
    metal_bridge::gemm_backward(
        (const float*)x_norm.raw_ptr(), (const float*)grad_q_proj_bf16.raw_ptr(), (float*)grad_Wq_bf16.raw_ptr(),
        hidden_dim, n_heads * head_dim, batch * seq_len);
    metal_bridge::gemm_backward(
        (const float*)x_norm.raw_ptr(), (const float*)grad_k_proj_bf16.raw_ptr(), (float*)grad_Wk_bf16.raw_ptr(),
        hidden_dim, n_kv_heads * head_dim, batch * seq_len);
    metal_bridge::gemm_backward(
        (const float*)x_norm.raw_ptr(), (const float*)grad_v_proj_bf16.raw_ptr(), (float*)grad_Wv_bf16.raw_ptr(),
        hidden_dim, n_kv_heads * head_dim, batch * seq_len);
  } else {
    grad_Wq = x_norm.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_q_proj.reshape({batch * seq_len, n_heads * head_dim}));
    grad_Wk = x_norm.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_k_proj.reshape({batch * seq_len, n_kv_heads * head_dim}));
    grad_Wv = x_norm.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_v_proj.reshape({batch * seq_len, n_kv_heads * head_dim}));
  }

  // 9. Compute gradient w.r.t normalized input: grad_x_norm = grad_q_proj @ Wq^T + grad_k_proj @ Wk^T + grad_v_proj @ Wv^T
  Tensor grad_x_norm({batch, seq_len, hidden_dim}, 0.0f, x_aligned.dtype());
  if (use_gpu) {
    metal_bridge::gemm_proj_trans_b(
        (const float*)grad_q_proj_bf16.raw_ptr(), (const float*)Wq_.raw_ptr(), (float*)grad_x_norm.raw_ptr(),
        batch * seq_len, hidden_dim, n_heads * head_dim);

    Tensor grad_k_norm({batch, seq_len, hidden_dim}, 0.0f, x_aligned.dtype());
    metal_bridge::gemm_proj_trans_b(
        (const float*)grad_k_proj_bf16.raw_ptr(), (const float*)Wk_.raw_ptr(), (float*)grad_k_norm.raw_ptr(),
        batch * seq_len, hidden_dim, n_kv_heads * head_dim);
    metal_bridge::residual_add((float*)grad_x_norm.raw_ptr(), (const float*)grad_k_norm.raw_ptr(), grad_x_norm.size());

    Tensor grad_v_norm({batch, seq_len, hidden_dim}, 0.0f, x_aligned.dtype());
    metal_bridge::gemm_proj_trans_b(
        (const float*)grad_v_proj_bf16.raw_ptr(), (const float*)Wv_.raw_ptr(), (float*)grad_v_norm.raw_ptr(),
        batch * seq_len, hidden_dim, n_kv_heads * head_dim);
    metal_bridge::residual_add((float*)grad_x_norm.raw_ptr(), (const float*)grad_v_norm.raw_ptr(), grad_x_norm.size());
  } else {
    grad_x_norm = grad_q_proj.matmul(Wq_.transpose());
    grad_x_norm.add_(grad_k_proj.matmul(Wk_.transpose()));
    grad_x_norm.add_(grad_v_proj.matmul(Wv_.transpose()));
  }

  // 10. Backpropagate through RMSNorm to get gradient w.r.t raw input x
  Tensor grad_weight_dummy({hidden_dim}, 0.0f, x_aligned.dtype());
  Tensor grad_x = rms_norm.backward(grad_x_norm, x_aligned, grad_weight_dummy);

  if (use_gpu) {
    if (grad_Wq.dtype() == DType::FP32) { Tensor tmp = grad_Wq_bf16.to_dtype(DType::FP32); std::memcpy(grad_Wq.data(), tmp.data(), tmp.raw_bytes()); }
    if (grad_Wk.dtype() == DType::FP32) { Tensor tmp = grad_Wk_bf16.to_dtype(DType::FP32); std::memcpy(grad_Wk.data(), tmp.data(), tmp.raw_bytes()); }
    if (grad_Wv.dtype() == DType::FP32) { Tensor tmp = grad_Wv_bf16.to_dtype(DType::FP32); std::memcpy(grad_Wv.data(), tmp.data(), tmp.raw_bytes()); }
    if (grad_Wo.dtype() == DType::FP32) { Tensor tmp = grad_Wo_bf16.to_dtype(DType::FP32); std::memcpy(grad_Wo.data(), tmp.data(), tmp.raw_bytes()); }
  }

  return use_gpu ? grad_x.to_dtype(grad_output.dtype()) : grad_x;
}
