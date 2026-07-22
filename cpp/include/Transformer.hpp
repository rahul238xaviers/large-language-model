#pragma once

#include "Attention.hpp"
#include "Positional.hpp"
#include "RMSNorm.hpp"
#include "Tensor.hpp"
#include "TransformerConfig.hpp"
#include <vector>

struct TransformerLayer {
  RMSNorm attn_norm;
  Attention attn;
  RMSNorm ffn_norm;

  // Individual weight tensors (for optimizer m/v tracking)
  Tensor w_gate;  // [H, I]
  Tensor w_up;    // [H, I]
  Tensor w_down;  // [I, H]

  // Fused weight tensors (for single-GEMM dispatch)
  Tensor w_qkv;      // [H, nH*HD + 2*nKV*HD] = [1024, 2048]
  Tensor w_gate_up;  // [H, 2*I] = [1024, 5504]
  ModelConfig cfg;   // stored config for helper methods

  TransformerLayer(const ModelConfig &config);
  const ModelConfig &config() const { return cfg; }

  // Fused forward helpers — dispatch one GEMM, return slices as pointer offsets
  void fused_qkv_forward(const float* input, float* output,
                          size_t B, size_t S) const;
  void fused_gate_up_forward(const float* input, float* gate_out,
                              float* up_out, size_t B, size_t S) const;

  Tensor backward(const Tensor &grad_output, const Tensor &h_in,
                  Tensor &grad_w_gate, Tensor &grad_w_up, Tensor &grad_w_down,
                  Tensor &grad_Wq, Tensor &grad_Wk, Tensor &grad_Wv,
                  Tensor &grad_Wo, const RoPE &rope,
                  KVCache *cache = nullptr) const;

  // ── Persistent gradient intermediates (allocated once, recycled per step) ──
  // Prevents 128/360 MB mmap + zero + munmap per layer per step.
  mutable Tensor grad_activated_;     // max shape: {B, S, I}
  mutable Tensor grad_gate_;          // max shape: {B, S, I}
  mutable Tensor grad_up_;            // max shape: {B, S, I}
  mutable Tensor grad_ffn_in_;        // max shape: {B, S, H}
  mutable Tensor grad_up_in_;         // max shape: {B, S, H}
};

class Transformer {
public:
  Transformer(const ModelConfig &config);

  Tensor forward(const Tensor &tokens, KVCache *cache = nullptr) const;
  void clear_h_cache() const { h_cache_.clear(); }
  const std::vector<Tensor> &h_cache() const { return h_cache_; }

  Tensor backward(const Tensor &grad_logits, const Tensor &tokens,
                  std::vector<Tensor> &grad_w_gate,
                  std::vector<Tensor> &grad_w_up,
                  std::vector<Tensor> &grad_w_down,
                  std::vector<Tensor> &grad_Wq, std::vector<Tensor> &grad_Wk,
                  std::vector<Tensor> &grad_Wv, std::vector<Tensor> &grad_Wo,
                  Tensor &grad_embeddings, Tensor &grad_output_projection,
                  const RoPE &rope) const;

  const Tensor &token_embeddings() const { return token_embeddings_; }
  Tensor &token_embeddings() { return token_embeddings_; }
  const Tensor &output_projection() const { return output_projection_; }
  Tensor &output_projection() { return output_projection_; }
  const std::vector<TransformerLayer> &layers() const { return layers_; }
  std::vector<TransformerLayer> &layers() { return layers_; }
  const RMSNorm &final_norm() const { return final_norm_; }
  RMSNorm &final_norm() { return final_norm_; }
  const ModelConfig &config() const { return config_; }

private:
  ModelConfig config_;
  Tensor token_embeddings_;
  std::vector<TransformerLayer> layers_;
  RMSNorm final_norm_;
  Tensor output_projection_;
  RoPE rope_;
  mutable std::vector<Tensor> h_cache_;
};
