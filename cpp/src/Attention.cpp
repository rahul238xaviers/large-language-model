#include "Attention.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

Attention::Attention(const ModelConfig &config)
    : config_(config),
      Wq_({config.hidden_dim, config.n_heads * config.head_dim}),
      Wk_({config.hidden_dim, config.n_kv_heads * config.head_dim}),
      Wv_({config.hidden_dim, config.n_kv_heads * config.head_dim}),
      Wo_({config.n_heads * config.head_dim, config.hidden_dim}) {};

Tensor matmul_project(const Tensor &x, const Tensor &w) {

  size_t batch_size = x.shape()[0];
  size_t seq_len = x.shape()[1];
  size_t hidden_dim = x.shape()[2];
  size_t out_dim = w.shape()[1];

  Tensor result({batch_size, seq_len, out_dim}, 0.0f);

  const float *x_data = x.data().data();
  const float *w_data = w.data().data();
  float *res_data = result.data().data();

  for (size_t b = 0; b < batch_size; ++b) {
    size_t x_batch_offset = b * seq_len * hidden_dim;
    size_t res_batch_offset = b * seq_len * out_dim;

    for (size_t s = 0; s < seq_len; ++s) {
      size_t x_row_offset = x_batch_offset + s * hidden_dim;
      size_t res_row_offset = res_batch_offset + s * out_dim;

      for (size_t d = 0; d < hidden_dim; ++d) {
        float val = x_data[x_row_offset + d];
        for (size_t o = 0; o < out_dim; ++o) {
          res_data[res_row_offset + o] += val * w_data[d * out_dim + o];
        }
      }
    }
  }

  return result;
}

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
}