#pragma once
#include "Positional.hpp"
#include "Tensor.hpp"
#include "TransformerConfig.hpp"
#include <vector>

class Attention {

public:
  Attention(const ModelConfig &config);
  Tensor forward(const Tensor &x, const RoPE &rope, KVCache *cache = nullptr,
                 size_t pos_offset = 0) const;
  const Tensor &Wq() const { return Wq_; }
  Tensor &Wq() { return Wq_; }
  const Tensor &Wk() const { return Wk_; }
  Tensor &Wk() { return Wk_; }
  const Tensor &Wv() const { return Wv_; }
  Tensor &Wv() { return Wv_; }
  const Tensor &Wo() const { return Wo_; }
  Tensor &Wo() { return Wo_; }

private:
  ModelConfig config_;
  Tensor Wq_;
  Tensor Wk_;
  Tensor Wv_;
  Tensor Wo_;
};