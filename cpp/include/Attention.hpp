/**
 * @file Attention.hpp
 * @brief Grouped Query Attention (GQA) layer declaration
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Attention computes context-weighted vectors from queries, keys, and values.
 * It supports Grouped Query Attention (GQA) where multiple query heads share
 * a single key/value head, reducing memory consumption and improving speed.
 */

#pragma once
#include "Positional.hpp"
#include "Tensor.hpp"
#include "TransformerConfig.hpp"

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

  Tensor backward(const Tensor &grad_output, const Tensor &x, const RoPE &rope,
                  Tensor &grad_Wq, Tensor &grad_Wk, Tensor &grad_Wv,
                  Tensor &grad_Wo, KVCache *cache = nullptr,
                  size_t pos_offset = 0) const;

private:
  ModelConfig config_;
  Tensor Wq_;
  Tensor Wk_;
  Tensor Wv_;
  Tensor Wo_;
};