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
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Construct a single transformer layer block
TransformerLayer::TransformerLayer(const ModelConfig &config)
    : attn_norm(config.hidden_dim, config.rms_norm_eps), attn(config),
      ffn_norm(config.hidden_dim, config.rms_norm_eps),
      w_gate({config.hidden_dim, config.intermediate_dim}, 0.0f),
      w_up({config.hidden_dim, config.intermediate_dim}, 0.0f),
      w_down({config.intermediate_dim, config.hidden_dim}, 0.0f) {}

// Construct the complete Transformer model
Transformer::Transformer(const ModelConfig &config)
    : config_(config),
      token_embeddings_({config.vocab_size, config.hidden_dim}, 0.0f),
      final_norm_(config.hidden_dim, config.rms_norm_eps),
      output_projection_({config.hidden_dim, config.vocab_size}, 0.0f),
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

  Tensor h = Tensor({batch_size, seq_len, config_.hidden_dim});

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t s = 0; s < seq_len; s++) {

      float token_id = tokens(b, s);
      if (token_id > config_.vocab_size - 1) {
        throw std::runtime_error("token_id is out of range");
      }
      size_t id = static_cast<size_t>(token_id);
      for (size_t d = 0; d < config_.hidden_dim; d++) {
        h(b, s, d) = token_embeddings_(id, d);
      }
    }
  }
  for (const auto &layer : layers_) {
    // 1. Attention Block with Residual
    Tensor attn_in = layer.attn_norm.forward(h);
    Tensor attn_out = layer.attn.forward(attn_in, rope_, cache);
    h.add_(attn_out);

    // 2. FFN Block (SwiGLU) with Residual
    Tensor ffn_in = layer.ffn_norm.forward(h);
    Tensor gate_proj = ffn_in.matmul(layer.w_gate);
    Tensor up_proj = ffn_in.matmul(layer.w_up);
    Tensor activated = activatations::swiglu(gate_proj, up_proj);
    Tensor ffn_out = activated.matmul(layer.w_down);
    h.add_(ffn_out);
  }

  // Final RMSNorm
  Tensor final_h = final_norm_.forward(h);

  // Project to vocabulary logits
  Tensor logits = final_h.matmul(output_projection_);

  return logits;
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
  for (size_t b = 0; b < batch; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      for (size_t i = 0; i < intermediate_dim; ++i) {
        float act_val = activated(b, s, i);
        for (size_t o = 0; o < hidden_dim; ++o) {
          grad_w_down(i, o) += act_val * grad_output(b, s, o);
        }
      }
    }
  }
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
  for (size_t b = 0; b < batch; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      for (size_t i = 0; i < hidden_dim; ++i) {
        float f_val = ffn_in(b, s, i);
        for (size_t o = 0; o < intermediate_dim; ++o) {
          grad_w_gate(i, o) += f_val * grad_gate(b, s, o);
          grad_w_up(i, o) += f_val * grad_up(b, s, o);
        }
      }
    }
  }
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

  // --- A. Recompute Forward States ---
  Tensor attn_in = attn_norm.forward(h_in);
  Tensor attn_out = attn.forward(attn_in, rope, cache);
  Tensor h_mid = h_in.add(attn_out);

  Tensor ffn_in = ffn_norm.forward(h_mid);
  Tensor gate_proj = ffn_in.matmul(w_gate);
  Tensor up_proj = ffn_in.matmul(w_up);
  Tensor activated = activatations::swiglu(gate_proj, up_proj);

  // Helper to transpose 2D weight matrices
  auto transpose_w = [](const Tensor &w) {
    size_t rows = w.shape()[0];
    size_t cols = w.shape()[1];
    Tensor transposed({cols, rows}, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        transposed(c, r) = w(r, c);
      }
    }
    return transposed;
  };

  // --- B. FFN Down Projection Backward ---
  accumulate_ffn_down_grads(batch, seq_len, intermediate_dim, hidden_dim,
                            activated, grad_output, grad_w_down);
  Tensor grad_activated = grad_output.matmul(transpose_w(w_down));

  // --- C. SwiGLU & Projection Backwards ---
  Tensor grad_gate(gate_proj.shape(), 0.0f);
  Tensor grad_up(up_proj.shape(), 0.0f);
  activatations::swiglu_backward(grad_activated, gate_proj, up_proj, grad_gate,
                                 grad_up);

  accumulate_ffn_gate_up_grads(batch, seq_len, hidden_dim, intermediate_dim,
                               ffn_in, grad_gate, grad_up, grad_w_gate,
                               grad_w_up);

  Tensor grad_ffn_in = grad_gate.matmul(transpose_w(w_gate));
  grad_ffn_in.add_(grad_up.matmul(transpose_w(w_up)));

  // --- D. FFN Norm & Residual Backward ---
  Tensor grad_ffn_norm_weight_dummy({hidden_dim}, 0.0f);
  Tensor grad_h_mid =
      ffn_norm.backward(grad_ffn_in, h_mid, grad_ffn_norm_weight_dummy);
  grad_h_mid.add_(grad_output);

  // --- E. Attention Layer & Norm & Residual Backward ---
  Tensor grad_attn_in = attn.backward(grad_h_mid, attn_in, rope, grad_Wq,
                                      grad_Wk, grad_Wv, grad_Wo, cache);

  Tensor grad_attn_norm_weight_dummy({hidden_dim}, 0.0f);
  Tensor grad_h_in =
      attn_norm.backward(grad_attn_in, h_in, grad_attn_norm_weight_dummy);
  grad_h_in.add_(grad_h_mid);

  return grad_h_in;
}

// Static helper to run a cached forward pass storing intermediate hidden states for layer-by-layer backprop
static std::vector<Tensor> run_forward_cache(
    const Transformer &model, const Tensor &tokens, const RoPE &rope, size_t hidden_dim) {
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

// Static helper to accumulate parameter gradients for the output projection layer
static void accumulate_output_projection_grads(
    size_t batch_size, size_t seq_len, size_t hidden_dim, size_t vocab_size,
    const Tensor &final_h, const Tensor &grad_logits, Tensor &grad_output_projection) {
  grad_output_projection.fill(0.0f);
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      for (size_t d = 0; d < hidden_dim; ++d) {
        float h_val = final_h(b, s, d);
        for (size_t v = 0; v < vocab_size; ++v) {
          grad_output_projection(d, v) += h_val * grad_logits(b, s, v);
        }
      }
    }
  }
}

// Static helper to accumulate gradients w.r.t token embeddings
static void accumulate_embedding_grads(
    size_t batch_size, size_t seq_len, size_t hidden_dim,
    const Tensor &tokens, const Tensor &grad_h, Tensor &grad_embeddings) {
  grad_embeddings.fill(0.0f);
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      size_t id = static_cast<size_t>(tokens(b, s));
      for (size_t d = 0; d < hidden_dim; ++d) {
        grad_embeddings(id, d) += grad_h(b, s, d);
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
 * It runs a single cached forward pass at the beginning to store intermediate hidden state outputs
 * for each layer, eliminating the memory and computation overhead of N layer re-evaluations.
 *
 * @param grad_logits Gradients w.r.t the logits output tensor of shape [batch_size, seq_len, vocab_size].
 * @param tokens Input token IDs tensor of shape [batch_size, seq_len].
 * @param grad_w_gate Vector of gate projection weight gradients per layer.
 * @param grad_w_up Vector of up projection weight gradients per layer.
 * @param grad_w_down Vector of down projection weight gradients per layer.
 * @param grad_Wq Vector of Query projection weight gradients per layer.
 * @param grad_Wk Vector of Key projection weight gradients per layer.
 * @param grad_Wv Vector of Value projection weight gradients per layer.
 * @param grad_Wo Vector of Output projection weight gradients per layer.
 * @param grad_embeddings Parameter gradient accumulator for the token embedding lookup matrix.
 * @param grad_output_projection Parameter gradient accumulator for the output vocabulary projection weights.
 * @param rope Rotary Position Embedding helper.
 * @return Tensor Input embedding gradients of shape [batch_size, seq_len, hidden_dim].
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

  // Helper to transpose 2D weight matrices
  auto transpose_w = [](const Tensor &w) {
    size_t rows = w.shape()[0];
    size_t cols = w.shape()[1];
    Tensor transposed({cols, rows}, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        transposed(c, r) = w(r, c);
      }
    }
    return transposed;
  };

  // --- 1. Forward Pass to cache intermediate hidden states ---
  std::vector<Tensor> h_states = run_forward_cache(*this, tokens, rope, hidden_dim);
  Tensor final_h = final_norm_.forward(h_states.back());

  // --- 2. Output Projection Backward ---
  accumulate_output_projection_grads(batch_size, seq_len, hidden_dim, vocab_size, final_h, grad_logits, grad_output_projection);

  Tensor grad_final_h = grad_logits.matmul(transpose_w(output_projection_));

  // --- 3. Final RMSNorm Backward ---
  Tensor grad_final_norm_weight_dummy({hidden_dim}, 0.0f);
  Tensor grad_h = final_norm_.backward(grad_final_h, h_states.back(),
                                       grad_final_norm_weight_dummy);

  // --- 4. Backprop through Stacked Layers (in reverse order) ---
  for (int l = static_cast<int>(config_.n_layers) - 1; l >= 0; --l) {
    grad_h = layers_[l].backward(grad_h, h_states[l], grad_w_gate[l],
                                 grad_w_up[l], grad_w_down[l], grad_Wq[l],
                                 grad_Wk[l], grad_Wv[l], grad_Wo[l], rope);
  }

  // --- 5. Embedding Lookup Backward ---
  accumulate_embedding_grads(batch_size, seq_len, hidden_dim, tokens, grad_h, grad_embeddings);

  return grad_h;
}
