/**
 * @file Checkpoint.hpp
 * @brief Safetensors-compatible checkpoint serialization and deserialization
 */

#pragma once

#include "Transformer.hpp"
#include "Optimizer.hpp"
#include <string>

class Checkpoint {
public:
  /**
   * @brief Serializes model weights and optimizer moments into safetensors files.
   *
   * Saves model weights to `filepath` and optimizer state to `filepath.opt.safetensors`.
   *
   * @param filepath Target file path for model weights.
   * @param model Reference to the Transformer model.
   * @param optimizer Reference to the parameter optimizer.
   * @param step Current training step index.
   * @return true If save succeeded.
   * @return false Otherwise.
   */
  static bool save(const std::string &filepath, Transformer &model, Optimizer &optimizer, size_t step);

  /**
   * @brief Deserializes model weights and optimizer moments from safetensors files.
   *
   * @param filepath Source file path for model weights.
   * @param model Reference to the Transformer model.
   * @param optimizer Reference to the parameter optimizer.
   * @param step Will be populated with the resumed training step index.
   * @return true If load succeeded.
   * @return false Otherwise.
   */
  static bool load(const std::string &filepath, Transformer &model, Optimizer &optimizer, size_t &step);
};
