/**
 * @file Transformer.cpp
 * @brief Implementation of the Transformer model and layer blocks forward &
 * backward passes
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Implements constructors, forward pass, and backward pass gradient propagation
 * operations for TransformerLayer and the Transformer network.
 */

#include "Transformer.hpp"
#include "Activations.hpp"
#include "Tensor.hpp"
#include "gpu_kernel/MetalBridge.hpp"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <limits>
#include <cmath>
#include <algorithm>

static void print_tensor_stats(const std::string &name, const Tensor &t) {
  if (t.size() == 0) {
    std::cout << "[STATS] " << name << " is empty" << std::endl;
    return;
  }
  Tensor t_fp32 = (t.dtype() == DType::BF16) ? t.to_dtype(DType::FP32) : t;
  const float *data = t_fp32.data();
  size_t n = t_fp32.size();
  
  float min_val = std::numeric_limits<float>::has_infinity ? std::numeric_limits<float>::infinity() : 1e38f;
  float max_val = std::numeric_limits<float>::has_infinity ? -std::numeric_limits<float>::infinity() : -1e38f;
  size_t nan_count = 0;
  size_t inf_count = 0;
  double sum = 0.0;
  
  for (size_t i = 0; i < n; ++i) {
    float val = data[i];
    if (std::isnan(val)) {
      nan_count++;
    } else if (std::isinf(val)) {
      inf_count++;
    } else {
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
      sum += val;
    }
  }
  
  std::cout << "[STATS] " << name << " | size: " << n 
            << " | min: " << min_val << " | max: " << max_val 
            << " | mean: " << (n > nan_count + inf_count ? sum / (n - nan_count - inf_count) : 0.0)
            << " | NaNs: " << nan_count << " | Infs: " << inf_count << std::endl;
}

// Construct a single transformer layer block
TransformerLayer::TransformerLayer(const ModelConfig &config)
    : attn_norm(config.hidden_dim, config.rms_norm_eps), attn(config),
      ffn_norm(config.hidden_dim, config.rms_norm_eps),
      w_gate({config.hidden_dim, config.intermediate_dim}, 0.0f, DType::BF16),
      w_up({config.hidden_dim, config.intermediate_dim}, 0.0f, DType::BF16),
      w_down({config.intermediate_dim, config.hidden_dim}, 0.0f, DType::BF16),
      cfg(config),
      // Fused QKV: [H, nH*HD + 2*nKV*HD] = [1024, 2048]
      w_qkv({config.hidden_dim, config.n_heads * config.head_dim + 2 * config.n_kv_heads * config.head_dim}, 0.0f, DType::BF16),
      // Fused gate+up: [H, 2*I] = [1024, 5504]
      w_gate_up({config.hidden_dim, 2 * config.intermediate_dim}, 0.0f, DType::BF16),
      grad_activated_({}, DType::BF16),
      grad_gate_({}, DType::BF16),
      grad_up_({}, DType::BF16),
      grad_ffn_in_({}, DType::BF16),
      grad_up_in_({}, DType::BF16) {
  // Copy individual weights into fused buffers
  // w_qkv = [Wq | Wk | Wv] where Wq=[H,H], Wk=[H,512], Wv=[H,512]
  // w_gate_up = [Wgate | Wup] where both are [H,I]
  // (Weights are initialized separately — this just allocates the fused views)
}

// ── Fused QKV forward: single GEMM, slice output by pointer offset ──
void TransformerLayer::fused_qkv_forward(const float* input, float* output,
                                           size_t B, size_t S) const {
  // input: [B*S, H], output: [B*S, 2048]
  // w_qkv: [H, 2048]
  // One GEMM call instead of three
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)(B*S), 2048, (int)config().hidden_dim,
              1.0f, input, (int)config().hidden_dim,
              w_qkv.data(), 2048, 0.0f, output, 2048);
  // Caller slices: Q at offset 0 (stride 1024), K at offset 1024 (stride 512),
  // V at offset 1536 (stride 512)
  // All three are in contiguous output[0..B*S*2048-1]
}

// ── Fused Gate+Up forward: single GEMM, two pointer views ──
void TransformerLayer::fused_gate_up_forward(const float* input, float* gate_out,
                                               float* up_out, size_t B, size_t S) const {
  size_t M = B * S;
  // input: [M, H], gate_out: [M, I], up_out: [M, I]
  // w_gate_up: [H, 2*I]
  // Allocate temp buffer for fused output
  std::vector<float> fused(M * 2 * config().intermediate_dim);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)M, (int)(2*config().intermediate_dim), (int)config().hidden_dim,
              1.0f, input, (int)config().hidden_dim,
              w_gate_up.data(), (int)(2*config().intermediate_dim),
              0.0f, fused.data(), (int)(2*config().intermediate_dim));
  // Slice: gate = fused[0..M*I-1], up = fused[M*I..2*M*I-1]
  memcpy(gate_out, fused.data(), M * config().intermediate_dim * sizeof(float));
  memcpy(up_out, fused.data() + M * config().intermediate_dim,
         M * config().intermediate_dim * sizeof(float));
}

// Construct the complete Transformer model
Transformer::Transformer(const ModelConfig &config)
    : config_(config),
      token_embeddings_({config.vocab_size, config.hidden_dim}, 0.0f, DType::BF16),
      final_norm_(config.hidden_dim, config.rms_norm_eps),
      output_projection_({config.hidden_dim, config.vocab_size}, 0.0f, DType::BF16),
      rope_(config.head_dim, config.max_seq_len, config.rope_base) {

  layers_.reserve(config.n_layers);
  for (size_t i = 0; i < config.n_layers; ++i) {
    layers_.emplace_back(config);
  }
}
Tensor Transformer::forward(const Tensor &tokens, KVCache *cache) const {
  if (tokens.shape().size() != 2) {
    throw std::runtime_error("Input tokens must be 2D (batch_size, seq_len)");
  }

  size_t batch_size = tokens.shape()[0];
  size_t seq_len = tokens.shape()[1];

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (h_cache_.size() != layers_.size() + 1 || h_cache_[0].shape() != Shape{batch_size, seq_len, config_.hidden_dim}) {
    h_cache_.clear();
    for (size_t i = 0; i <= layers_.size(); ++i) {
      h_cache_.emplace_back(Shape{batch_size, seq_len, config_.hidden_dim}, 0.0f, use_gpu ? DType::BF16 : DType::FP32);
    }
  }

  Tensor &h_0 = h_cache_[0];

  if (use_gpu) {
    size_t total_tokens = batch_size * seq_len;
    std::vector<uint32_t> tokens_uint32(total_tokens);
    for (size_t i = 0; i < total_tokens; ++i) {
      tokens_uint32[i] = static_cast<uint32_t>(tokens.data()[i]);
    }
    metal_bridge::embedding_forward(
        tokens_uint32.data(),
        token_embeddings_.data(),
        h_0.data(),
        total_tokens,
        config_.hidden_dim,
        config_.vocab_size
    );
  } else {
    for (size_t b = 0; b < batch_size; b++) {
      for (size_t s = 0; s < seq_len; s++) {
        float token_id = tokens(b, s);
        if (token_id > config_.vocab_size - 1) {
          throw std::runtime_error("token_id is out of range");
        }
        size_t id = static_cast<size_t>(token_id);
        for (size_t d = 0; d < config_.hidden_dim; d++) {
          h_0(b, s, d) = token_embeddings_(id, d);
        }
      }
    }
  }

  // Per-layer forward loop
  for (size_t li = 0; li < layers_.size(); ++li) {
    const auto &layer = layers_[li];
    const Tensor &h_cur = h_cache_[li];
    Tensor &h_next = h_cache_[li + 1];

    Tensor attn_in = layer.attn_norm.forward(h_cur);
    Tensor attn_out = layer.attn.forward(attn_in, rope_, cache);

    if (use_gpu) {
      // Copy h_cur to h_next asynchronously on GPU so h_cur remains untouched for backward pass
      metal_bridge::copy_buffer_async(h_next.data(), h_next.raw_bytes(), h_cur.data(), h_cur.raw_bytes());

      // Fused: h_next += attn_out  (in-place on h_next)  AND  ffn_in = rmsnorm(h_next)
      Tensor ffn_in({batch_size, seq_len, config_.hidden_dim}, DType::BF16);
      metal_bridge::fused_add_norm(h_next.data(), (const float*)attn_out.raw_ptr(),
                                   (const float*)layer.ffn_norm.weight().raw_ptr(),
                                   (float*)ffn_in.raw_ptr(),
                                   batch_size * seq_len,
                                   config_.hidden_dim,
                                   config_.rms_norm_eps);

      Tensor gate_proj = ffn_in.matmul(layer.w_gate);
      Tensor up_proj = ffn_in.matmul(layer.w_up);
      Tensor ffn_out({batch_size, seq_len, config_.hidden_dim}, DType::BF16);
      metal_bridge::fused_swiglu_gemm(
          (const float*)gate_proj.raw_ptr(), (const float*)up_proj.raw_ptr(), (const float*)layer.w_down.data(),
          (float*)ffn_out.raw_ptr(),
          batch_size * seq_len, config_.hidden_dim, config_.intermediate_dim);

      metal_bridge::residual_add(h_next.data(), (const float*)ffn_out.raw_ptr(), h_next.size());
    } else {
      h_next = h_cur;
      h_next.add_(attn_out);
      Tensor ffn_in = layer.ffn_norm.forward(h_next);
      Tensor activated = activatations::swiglu(ffn_in.matmul(layer.w_gate),
                                                ffn_in.matmul(layer.w_up));
      Tensor ffn_out = activated.matmul(layer.w_down);
      h_next.add_(ffn_out);
    }
  }

  // Final RMSNorm
  Tensor final_h = final_norm_.forward(h_cache_.back());

  if (getenv("DEBUG_LOGITS")) {
    auto ncnt = [](const uint16_t* p, size_t n) { size_t c = 0; for (size_t i = 0; i < n; ++i) if ((p[i] & 0x7F80) == 0x7F80) c++; return c; };
    printf("[F] h NaN=%zu, final_h NaN=%zu\n",
           ncnt((const uint16_t*)h_cache_.back().raw_ptr(), h_cache_.back().num_elements()),
           ncnt((const uint16_t*)final_h.raw_ptr(), final_h.num_elements()));
  }

  // Project to vocabulary logits — reuse the persistent logits_ buffer so the
  // 6.5 GB allocation happens once, not every step (avoids page-fault churn).
  Shape logits_shape = h_cache_.back().shape();
  logits_shape.dims[2] = config_.vocab_size;
  logits_shape.ndim = 3;
  if (logits_.shape() != logits_shape)
    logits_ = Tensor(logits_shape, 0.0f, DType::BF16);

  metal_bridge::gemm_bf16(final_h.data(), output_projection_.data(),
                          (float*)logits_.raw_ptr(),
                          batch_size * seq_len, config_.vocab_size,
                          config_.hidden_dim, false, false, true);
  return logits_;
}

/**
 * @brief Static helper to accumulate gradients for the FFN down projection
 * layer.
 *
 * Implements outer product multiplication of FFN activated states and
 * downstream gradients.
 *
 * @param batch Batch size.
 * @param seq_len Sequence length.
 * @param intermediate_dim FFN intermediate dimension.
 * @param hidden_dim Hidden state dimension.
 * @param activated SwiGLU activation output.
 * @param grad_output Downstream FFN block output gradients.
 * @param grad_w_down Parameter gradient accumulator for down projection
 * weights.
 */
static void
accumulate_ffn_down_grads(size_t batch, size_t seq_len, size_t intermediate_dim,
                          size_t hidden_dim, const Tensor &activated,
                          const Tensor &grad_output, Tensor &grad_w_down) {
  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      metal_bridge::gemm_backward(
          (const float*)activated.raw_ptr(),
          (const float*)grad_output.raw_ptr(),
          (float*)grad_w_down.raw_ptr(),
          intermediate_dim,
          hidden_dim,
          batch * seq_len
      );
      return;
    }
  }

  // CPU fallback
  grad_w_down.add_(activated.reshape({batch * seq_len, intermediate_dim}).transpose().matmul(grad_output.reshape({batch * seq_len, hidden_dim})));
}

/**
 * @brief Static helper to accumulate gradients for the FFN gate and up
 * projection layers.
 *
 * Implements outer product multiplication of FFN input states and downstream
 * projection gradients.
 *
 * @param batch Batch size.
 * @param seq_len Sequence length.
 * @param hidden_dim Hidden state dimension.
 * @param intermediate_dim FFN intermediate dimension.
 * @param ffn_in Input activations to the FFN block.
 * @param grad_gate Downstream gradients w.r.t the SwiGLU gate projection.
 * @param grad_up Downstream gradients w.r.t the SwiGLU up projection.
 * @param grad_w_gate Parameter gradient accumulator for gate projection
 * weights.
 * @param grad_w_up Parameter gradient accumulator for up projection weights.
 */
static void
accumulate_ffn_gate_up_grads(size_t batch, size_t seq_len, size_t hidden_dim,
                             size_t intermediate_dim, const Tensor &ffn_in,
                             const Tensor &grad_gate, const Tensor &grad_up,
                             Tensor &grad_w_gate, Tensor &grad_w_up) {
  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      metal_bridge::gemm_backward(
          (const float*)ffn_in.raw_ptr(),
          (const float*)grad_gate.raw_ptr(),
          (float*)grad_w_gate.raw_ptr(),
          hidden_dim,
          intermediate_dim,
          batch * seq_len
      );
      metal_bridge::gemm_backward(
          (const float*)ffn_in.raw_ptr(),
          (const float*)grad_up.raw_ptr(),
          (float*)grad_w_up.raw_ptr(),
          hidden_dim,
          intermediate_dim,
          batch * seq_len
      );
      return;
    }
  }

  // CPU fallback
  grad_w_gate.add_(ffn_in.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_gate.reshape({batch * seq_len, intermediate_dim})));
  grad_w_up.add_(ffn_in.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_up.reshape({batch * seq_len, intermediate_dim})));
}

/**
 * @brief Performs the backward pass of a single Transformer layer block.
 *
 * Implements backpropagation through FFN block, SwiGLU activations, GQA
 * self-attention, RMS normalization, and residual layers. Recomputes
 * intermediate forward states on the fly to keep memory overhead to zero.
 *
 * @param grad_output Gradient w.r.t layer output hidden states of shape [batch,
 * seq_len, hidden_dim].
 * @param h_in Input hidden states to this layer of shape [batch, seq_len,
 * hidden_dim].
 * @param grad_w_gate Gradient w.r.t FFN gate weights of shape [hidden_dim,
 * intermediate_dim].
 * @param grad_w_up Gradient w.r.t FFN up projection weights of shape
 * [hidden_dim, intermediate_dim].
 * @param grad_w_down Gradient w.r.t FFN down projection weights of shape
 * [intermediate_dim, hidden_dim].
 * @param grad_Wq Gradient w.r.t Attention Query weights of shape [hidden_dim,
 * n_heads * head_dim].
 * @param grad_Wk Gradient w.r.t Attention Key weights of shape [hidden_dim,
 * n_kv_heads * head_dim].
 * @param grad_Wv Gradient w.r.t Attention Value weights of shape [hidden_dim,
 * n_kv_heads * head_dim].
 * @param grad_Wo Gradient w.r.t Attention Output projection weights of shape
 * [n_heads * head_dim, hidden_dim].
 * @param rope Rotary Position Embedding helper.
 * @param cache Optional pointer to Key-Value Cache.
 * @return Tensor Input gradient w.r.t layer input h_in of shape [batch,
 * seq_len, hidden_dim].
 */
Tensor TransformerLayer::backward(const Tensor &grad_output, const Tensor &h_in,
                                  Tensor &grad_w_gate, Tensor &grad_w_up,
                                  Tensor &grad_w_down, Tensor &grad_Wq,
                                  Tensor &grad_Wk, Tensor &grad_Wv,
                                  Tensor &grad_Wo, const RoPE &rope,
                                  KVCache *cache) const {
  if (h_in.shape().size() != 3) {
    throw std::invalid_argument(
        "Input hidden states h_in must have exactly 3 dimensions");
  }
  if (grad_output.shape() != h_in.shape()) {
    throw std::invalid_argument("grad_output shape must match h_in shape");
  }
  if (grad_w_gate.shape() != w_gate.shape()) {
    throw std::invalid_argument("grad_w_gate shape must match w_gate shape");
  }
  if (grad_w_up.shape() != w_up.shape()) {
    throw std::invalid_argument("grad_w_up shape must match w_up shape");
  }
  if (grad_w_down.shape() != w_down.shape()) {
    throw std::invalid_argument("grad_w_down shape must match w_down shape");
  }
  if (grad_Wq.shape() != attn.Wq().shape()) {
    throw std::invalid_argument("grad_Wq shape must match Wq_ shape");
  }
  if (grad_Wk.shape() != attn.Wk().shape()) {
    throw std::invalid_argument("grad_Wk shape must match Wk_ shape");
  }
  if (grad_Wv.shape() != attn.Wv().shape()) {
    throw std::invalid_argument("grad_Wv shape must match Wv_ shape");
  }
  if (grad_Wo.shape() != attn.Wo().shape()) {
    throw std::invalid_argument("grad_Wo shape must match Wo_ shape");
  }

  size_t batch = h_in.shape()[0];
  size_t seq_len = h_in.shape()[1];
  size_t hidden_dim = h_in.shape()[2];
  size_t intermediate_dim = w_gate.shape()[1];

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  // --- A. Recompute Forward States (BF16 if GPU is enabled) ---
  Tensor h_in_aligned = use_gpu ? h_in.to_dtype(DType::BF16) : h_in;
  Tensor grad_output_aligned = use_gpu ? grad_output.to_dtype(DType::BF16) : grad_output;

  if (use_gpu) {
    if (grad_w_gate_bf16_.shape() != grad_w_gate.shape()) grad_w_gate_bf16_ = Tensor(grad_w_gate.shape(), 0.0f, DType::BF16);
    if (grad_w_up_bf16_.shape() != grad_w_up.shape())     grad_w_up_bf16_ = Tensor(grad_w_up.shape(), 0.0f, DType::BF16);
    if (grad_w_down_bf16_.shape() != grad_w_down.shape()) grad_w_down_bf16_ = Tensor(grad_w_down.shape(), 0.0f, DType::BF16);
  }
  Tensor &grad_w_gate_bf16 = use_gpu ? grad_w_gate_bf16_ : grad_w_gate;
  Tensor &grad_w_up_bf16 = use_gpu ? grad_w_up_bf16_ : grad_w_up;
  Tensor &grad_w_down_bf16 = use_gpu ? grad_w_down_bf16_ : grad_w_down;

  attn_in_ = attn_norm.forward(h_in_aligned);
  attn_out_ = attn.forward(attn_in_, rope, cache);
  h_mid_ = h_in_aligned.add(attn_out_);

  ffn_in_ = ffn_norm.forward(h_mid_);
  gate_proj_ = ffn_in_.matmul(w_gate);
  up_proj_ = ffn_in_.matmul(w_up);
  activated_ = activatations::swiglu(gate_proj_, up_proj_);

  // --- B. FFN Down Projection Backward ---
  accumulate_ffn_down_grads(batch, seq_len, intermediate_dim, hidden_dim,
                            activated_, grad_output_aligned, grad_w_down_bf16);

  // Reuse persistent storage — no allocation, no zeroing
  grad_activated_.resize_storage({batch, seq_len, intermediate_dim});
  if (use_gpu) {
    metal_bridge::gemm_proj_trans_b((const float*)grad_output_aligned.raw_ptr(), (const float*)w_down.raw_ptr(),
                                    (float*)grad_activated_.raw_ptr(), batch * seq_len,
                                    intermediate_dim, hidden_dim);
  } else {
    auto transpose_w = [](const Tensor &w) {
      size_t rows = w.shape()[0];
      size_t cols = w.shape()[1];
      Tensor transposed({cols, rows}, 0.0f);
      vDSP_mtrans(w.data(), 1, transposed.data(), 1,
                  static_cast<vDSP_Length>(cols), static_cast<vDSP_Length>(rows));
      return transposed;
    };
    grad_activated_ = grad_output_aligned.matmul(transpose_w(w_down));
  }

  // --- C. SwiGLU & Projection Backwards ---
  // Reuse persistent storage — no alloc, GPU-blit zero (async, tracked by FIFO)
  grad_gate_.resize_storage(gate_proj_.shape());
  grad_up_.resize_storage(up_proj_.shape());
  if (use_gpu) {
    metal_bridge::fill_zero_async(grad_gate_.data(), grad_gate_.raw_bytes());
    metal_bridge::fill_zero_async(grad_up_.data(), grad_up_.raw_bytes());
  }
  activatations::swiglu_backward(grad_activated_, gate_proj_, up_proj_,
                                  grad_gate_, grad_up_);

  accumulate_ffn_gate_up_grads(batch, seq_len, hidden_dim, intermediate_dim,
                               ffn_in_, grad_gate_, grad_up_, grad_w_gate_bf16,
                               grad_w_up_bf16);

  grad_ffn_in_.resize_storage({batch, seq_len, hidden_dim});
  if (use_gpu) {
    metal_bridge::gemm_proj_trans_b((const float*)grad_gate_.raw_ptr(), (const float*)w_gate.raw_ptr(),
                                    (float*)grad_ffn_in_.raw_ptr(), batch * seq_len,
                                    hidden_dim, intermediate_dim);
    grad_up_in_.resize_storage({batch, seq_len, hidden_dim});
    metal_bridge::gemm_proj_trans_b((const float*)grad_up_.raw_ptr(), (const float*)w_up.raw_ptr(),
                                    (float*)grad_up_in_.raw_ptr(), batch * seq_len,
                                    hidden_dim, intermediate_dim);
    metal_bridge::residual_add((float*)grad_ffn_in_.raw_ptr(), (const float*)grad_up_in_.raw_ptr(), grad_ffn_in_.size());
  } else {
    auto transpose_w = [](const Tensor &w) {
      size_t rows = w.shape()[0];
      size_t cols = w.shape()[1];
      Tensor transposed({cols, rows}, 0.0f);
      vDSP_mtrans(w.data(), 1, transposed.data(), 1,
                  static_cast<vDSP_Length>(cols), static_cast<vDSP_Length>(rows));
      return transposed;
    };
    grad_ffn_in_ = grad_gate_.matmul(transpose_w(w_gate));
    grad_ffn_in_.add_(grad_up_.matmul(transpose_w(w_up)));
  }

  // --- D. FFN Norm & Residual Backward (FUSED) ---
  if (grad_h_mid_.shape() != Shape{batch, seq_len, hidden_dim} || grad_h_mid_.dtype() != DType::BF16) {
    grad_h_mid_ = Tensor({batch, seq_len, hidden_dim}, 0.0f, DType::BF16);
  }
  if (use_gpu) {
    metal_bridge::fused_backward_add_norm(
        (const float*)grad_ffn_in_.raw_ptr(), (const float*)h_mid_.raw_ptr(),
        (const float*)ffn_norm.weight().raw_ptr(), (const float*)grad_output_aligned.raw_ptr(),
        (float*)grad_h_mid_.raw_ptr(),
        batch * seq_len, hidden_dim, cfg.rms_norm_eps);
  } else {
    Tensor grad_ffn_norm_weight_dummy({hidden_dim}, 0.0f);
    grad_h_mid_ = ffn_norm.backward(grad_ffn_in_, h_mid_, grad_ffn_norm_weight_dummy);
    grad_h_mid_.add_(grad_output_aligned);
  }

  // --- E. Attention Layer & Norm & Residual Backward ---
  grad_attn_in_ = attn.backward(grad_h_mid_, attn_in_, rope, grad_Wq,
                                grad_Wk, grad_Wv, grad_Wo, cache);

  // --- F. Attention Norm & Residual Backward (FUSED) ---
  if (grad_h_in_.shape() != Shape{batch, seq_len, hidden_dim} || grad_h_in_.dtype() != DType::BF16) {
    grad_h_in_ = Tensor({batch, seq_len, hidden_dim}, 0.0f, DType::BF16);
  }
  if (use_gpu) {
    metal_bridge::fused_backward_add_norm(
        (const float*)grad_attn_in_.raw_ptr(), (const float*)h_in_aligned.raw_ptr(),
        (const float*)attn_norm.weight().raw_ptr(), (const float*)grad_h_mid_.raw_ptr(),
        (float*)grad_h_in_.raw_ptr(),
        batch * seq_len, hidden_dim, cfg.rms_norm_eps);
  } else {
    Tensor grad_attn_norm_weight_dummy({hidden_dim}, 0.0f);
    grad_h_in_ = attn_norm.backward(grad_attn_in_, h_in_aligned, grad_attn_norm_weight_dummy);
    grad_h_in_.add_(grad_h_mid_);
  }

  if (use_gpu) {
    if (grad_w_gate.dtype() == DType::FP32) { metal_bridge::convert_bf16_to_fp32((const float*)grad_w_gate_bf16.raw_ptr(), (float*)grad_w_gate.raw_ptr(), grad_w_gate.size()); }
    if (grad_w_up.dtype() == DType::FP32) { metal_bridge::convert_bf16_to_fp32((const float*)grad_w_up_bf16.raw_ptr(), (float*)grad_w_up.raw_ptr(), grad_w_up.size()); }
    if (grad_w_down.dtype() == DType::FP32) { metal_bridge::convert_bf16_to_fp32((const float*)grad_w_down_bf16.raw_ptr(), (float*)grad_w_down.raw_ptr(), grad_w_down.size()); }
  }

  return use_gpu ? grad_h_in_.to_dtype(grad_output.dtype()) : grad_h_in_;
}

// Static helper to run a cached forward pass storing intermediate hidden states
// for layer-by-layer backprop
static std::vector<Tensor> run_forward_cache(const Transformer &model,
                                             const Tensor &tokens,
                                             const RoPE &rope,
                                             size_t hidden_dim) {
  size_t batch_size = tokens.shape()[0];
  size_t seq_len = tokens.shape()[1];

  std::vector<Tensor> h_states;
  Tensor h({batch_size, seq_len, hidden_dim});

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t s = 0; s < seq_len; s++) {
      size_t id = static_cast<size_t>(tokens(b, s));
      for (size_t d = 0; d < hidden_dim; d++) {
        h(b, s, d) = model.token_embeddings()(id, d);
      }
    }
  }

  h_states.push_back(h);
  for (const auto &layer : model.layers()) {
    Tensor attn_in = layer.attn_norm.forward(h);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    h.add_(attn_out);

    Tensor ffn_in = layer.ffn_norm.forward(h);
    Tensor gate_proj = ffn_in.matmul(layer.w_gate);
    Tensor up_proj = ffn_in.matmul(layer.w_up);
    Tensor activated = activatations::swiglu(gate_proj, up_proj);
    Tensor ffn_out = activated.matmul(layer.w_down);
    h.add_(ffn_out);

    h_states.push_back(h);
  }
  return h_states;
}

static void accumulate_output_projection_grads(
    size_t batch_size, size_t seq_len, size_t hidden_dim, size_t vocab_size,
    const Tensor &final_h, const Tensor &grad_logits,
    Tensor &grad_output_projection, Tensor &grad_output_projection_bf16) {
  // grad_output_projection [H, V] = final_h[B*S, H].T @ grad_logits[B*S, V]
  // Use gemm_backward (trA=true) so both inputs stay BF16.
  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (use_gpu) {
    if (grad_output_projection_bf16.shape() != Shape{hidden_dim, vocab_size}) {
      grad_output_projection_bf16 = Tensor({hidden_dim, vocab_size}, 0.0f, DType::BF16);
    }
    metal_bridge::gemm_backward(
        (const float*)final_h.raw_ptr(),
        (const float*)grad_logits.raw_ptr(),
        (float*)grad_output_projection_bf16.raw_ptr(),
        hidden_dim, vocab_size, batch_size * seq_len);
    metal_bridge::convert_bf16_to_fp32(
        (const float*)grad_output_projection_bf16.raw_ptr(),
        (float*)grad_output_projection.raw_ptr(),
        grad_output_projection.size());
  } else {
    metal_bridge::gemm_backward(
        (const float*)final_h.raw_ptr(),
        (const float*)grad_logits.raw_ptr(),
        (float*)grad_output_projection.raw_ptr(),
        hidden_dim, vocab_size, batch_size * seq_len);
  }
}

// Static helper to accumulate gradients w.r.t token embeddings
static void accumulate_embedding_grads(size_t batch_size, size_t seq_len,
                                       size_t hidden_dim, const Tensor &tokens,
                                       const Tensor &grad_h,
                                       Tensor &grad_embeddings) {
  grad_embeddings.fill(0.0f);
  // grad_h is BF16 (2-byte packed) in the GPU path; Tensor::operator() reads via
  // data_float() (4 bytes/elem), which would over-read 2x past the buffer and
  // return garbage.  Convert to FP32 first so the indexed reads are in-bounds.
  Tensor gh = (grad_h.dtype() == DType::BF16) ? grad_h.to_dtype(DType::FP32)
                                              : grad_h;
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      size_t id = static_cast<size_t>(tokens(b, s));
      for (size_t d = 0; d < hidden_dim; ++d) {
        grad_embeddings(id, d) += gh(b, s, d);
      }
    }
  }
}

/**
 * @brief Performs the backward pass of the entire Transformer model.
 *
 * Implements reverse-mode gradient propagation through:
 * 1. Logit projection matrix.
 * 2. Final RMS normalization layer.
 * 3. Stack of Transformer layer blocks (in reverse sequence).
 * 4. Token embeddings table.
 *
 * It runs a single cached forward pass at the beginning to store intermediate
 * hidden state outputs for each layer, eliminating the memory and computation
 * overhead of N layer re-evaluations.
 *
 * @param grad_logits Gradients w.r.t the logits output tensor of shape
 * [batch_size, seq_len, vocab_size].
 * @param tokens Input token IDs tensor of shape [batch_size, seq_len].
 * @param grad_w_gate Vector of gate projection weight gradients per layer.
 * @param grad_w_up Vector of up projection weight gradients per layer.
 * @param grad_w_down Vector of down projection weight gradients per layer.
 * @param grad_Wq Vector of Query projection weight gradients per layer.
 * @param grad_Wk Vector of Key projection weight gradients per layer.
 * @param grad_Wv Vector of Value projection weight gradients per layer.
 * @param grad_Wo Vector of Output projection weight gradients per layer.
 * @param grad_embeddings Parameter gradient accumulator for the token embedding
 * lookup matrix.
 * @param grad_output_projection Parameter gradient accumulator for the output
 * vocabulary projection weights.
 * @param rope Rotary Position Embedding helper.
 * @return Tensor Input embedding gradients of shape [batch_size, seq_len,
 * hidden_dim].
 */

Tensor Transformer::backward(
    const Tensor &grad_logits, const Tensor &tokens,
    std::vector<Tensor> &grad_w_gate, std::vector<Tensor> &grad_w_up,
    std::vector<Tensor> &grad_w_down, std::vector<Tensor> &grad_Wq,
    std::vector<Tensor> &grad_Wk, std::vector<Tensor> &grad_Wv,
    std::vector<Tensor> &grad_Wo, Tensor &grad_embeddings,
    Tensor &grad_output_projection, const RoPE &rope) const {
  size_t batch_size = tokens.shape()[0];
  size_t seq_len = tokens.shape()[1];
  size_t hidden_dim = config_.hidden_dim;
  size_t vocab_size = config_.vocab_size;

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  bool profile_bwd = getenv("PROFILE_BWD") != nullptr;
  bool skip_outproj_bwd = getenv("DEBUG_SKIP_OUTPROJ_BWD") != nullptr;
  auto tbwd_start = std::chrono::high_resolution_clock::now();
  if (profile_bwd) std::cout << "[PROFILE-BWD] Starting Model Backward..." << std::endl;

  // --- 1. Use cached hidden states from the forward pass ---
  const auto &h_states = h_cache_;
  final_h_ = final_norm_.forward(h_states.back());

  // --- 2. Output Projection Backward ---
  if (profile_bwd) std::cout << "[PROFILE-BWD]   1. Output Projection Grad..." << std::endl;
  auto top_start = std::chrono::high_resolution_clock::now();
  Tensor grad_h;
  if (!skip_outproj_bwd) {
    accumulate_output_projection_grads(batch_size, seq_len, hidden_dim,
                                       vocab_size, final_h_, grad_logits,
                                       grad_output_projection, grad_output_projection_bf16_);

    Tensor grad_final_h({batch_size, seq_len, hidden_dim}, 0.0f, DType::BF16);
    if (use_gpu) {
      if (grad_final_h_bf16_.shape() != Shape{batch_size, seq_len, hidden_dim}) {
        grad_final_h_bf16_ = Tensor({batch_size, seq_len, hidden_dim}, 0.0f, DType::BF16);
      }
      metal_bridge::gemm_proj_trans_b(
          grad_logits.data(),
          output_projection_.data(),
          (float*)grad_final_h_bf16_.raw_ptr(),
          batch_size * seq_len,
          hidden_dim,
          vocab_size
      );
      grad_final_h = grad_final_h_bf16_;
    } else {
      auto transpose_w = [](const Tensor &w) {
        size_t rows = w.shape()[0];
        size_t cols = w.shape()[1];
        Tensor transposed({cols, rows}, 0.0f);
        vDSP_mtrans(w.data(), 1, transposed.data(), 1,
                    static_cast<vDSP_Length>(cols), static_cast<vDSP_Length>(rows));
        return transposed;
      };
      grad_final_h = grad_logits.matmul(transpose_w(output_projection_));
    }

    // --- 3. Final RMSNorm Backward ---
    if (profile_bwd) std::cout << "[PROFILE-BWD]   2. Final RMSNorm Backward..." << std::endl;
    Tensor grad_final_norm_weight_dummy({hidden_dim}, 0.0f);
    grad_h = final_norm_.backward(grad_final_h, h_states.back(),
                                       grad_final_norm_weight_dummy);
    if (getenv("DEBUG_STATS")) {
      print_tensor_stats("DEBUG final_h_", final_h_);
      print_tensor_stats("DEBUG h_states.back()", h_states.back());
      print_tensor_stats("DEBUG grad_final_h", grad_final_h);
      print_tensor_stats("DEBUG grad_h after final_norm", grad_h);
    }
  } else {
    grad_h = final_h_;
  }
  if (profile_bwd) {
    auto top_end = std::chrono::high_resolution_clock::now();
    double ms_op = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(top_end - top_start).count()) / 1000.0;
    std::cout << "[PROFILE-BWD]   outproj+grad_final_h+finalnorm finished in " << ms_op << " ms" << std::endl;
  }

  // --- 4. Backprop through Stacked Layers (in reverse order) ---
  if (profile_bwd) std::cout << "[PROFILE-BWD]   3. Layer-by-Layer Backward (" << config_.n_layers << " layers)..." << std::endl;
  auto tl_start = std::chrono::high_resolution_clock::now();
  grad_h_layers_.resize(config_.n_layers + 1);
  grad_h_layers_[config_.n_layers] = grad_h;
  for (int l = static_cast<int>(config_.n_layers) - 1; l >= 0; --l) {
    grad_h_layers_[l] = layers_[l].backward(grad_h_layers_[l + 1], h_states[l], grad_w_gate[l],
                                            grad_w_up[l], grad_w_down[l], grad_Wq[l],
                                            grad_Wk[l], grad_Wv[l], grad_Wo[l], rope);
    if (getenv("DEBUG_STATS")) {
      if (use_gpu) {
        metal_bridge::end_scope();
      }
      
      std::string lstr = std::to_string(l);
      print_tensor_stats("DEBUG grad_h_layers_[" + lstr + "]", grad_h_layers_[l]);
      
      Tensor check = (grad_h_layers_[l].dtype() == DType::BF16) ? grad_h_layers_[l].to_dtype(DType::FP32) : grad_h_layers_[l];
      const float* ptr = check.data();
      bool found_nan = false;
      for (size_t i = 0; i < check.size(); ++i) {
        if (std::isnan(ptr[i])) {
          found_nan = true;
          break;
        }
      }
      
      if (found_nan) {
        std::cerr << "\n!!! NAN DETECTED AT LAYER " << l << " !!!\n"
                  << "Inputs to this layer's backward pass:\n";
        print_tensor_stats("  - grad_output (grad_h_layers_[" + std::to_string(l+1) + "])", grad_h_layers_[l+1]);
        print_tensor_stats("  - h_in (h_states[" + std::to_string(l) + "])", h_states[l]);
        std::cerr << "Layer parameters:\n";
        print_tensor_stats("  - w_gate", layers_[l].w_gate);
        print_tensor_stats("  - w_up", layers_[l].w_up);
        print_tensor_stats("  - w_down", layers_[l].w_down);
        print_tensor_stats("  - attn.Wq", layers_[l].attn.Wq());
        print_tensor_stats("  - attn.Wk", layers_[l].attn.Wk());
        print_tensor_stats("  - attn.Wv", layers_[l].attn.Wv());
        print_tensor_stats("  - attn.Wo", layers_[l].attn.Wo());
        std::cerr << std::endl;
      }
      
      if (use_gpu) {
        metal_bridge::begin_scope();
      }
    }
  }
  if (profile_bwd) {
    auto tl_end = std::chrono::high_resolution_clock::now();
    double ms_layers = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(tl_end - tl_start).count()) / 1000.0;
    std::cout << "[PROFILE-BWD]   3. All " << config_.n_layers << " layers finished in " << ms_layers << " ms" << std::endl;
  }

  // --- 5. Embedding Lookup Backward ---
  if (profile_bwd) std::cout << "[PROFILE-BWD]   4. Embedding Grad Accumulation..." << std::endl;
  auto temb_start = std::chrono::high_resolution_clock::now();
  if (use_gpu) {
    metal_bridge::end_scope();
  }
  accumulate_embedding_grads(batch_size, seq_len, hidden_dim, tokens, grad_h_layers_[0],
                             grad_embeddings);

  if (profile_bwd) {
    auto temb_end = std::chrono::high_resolution_clock::now();
    double ms_emb = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(temb_end - temb_start).count()) / 1000.0;
    std::cout << "[PROFILE-BWD]   embedding grad finished in " << ms_emb << " ms" << std::endl;
    auto tbwd_end = std::chrono::high_resolution_clock::now();
    double ms_bwd_total = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(tbwd_end - tbwd_start).count()) / 1000.0;
    std::cout << "[PROFILE-BWD] Finished Model Backward in " << ms_bwd_total << " ms" << std::endl;
  }

  return grad_h_layers_[0];
}
