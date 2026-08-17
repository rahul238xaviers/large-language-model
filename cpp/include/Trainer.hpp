/**
 * @file Trainer.hpp
 * @brief Top-level orchestrator managing neural network pre-training loop
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Manages standard pre-training iteration loops:
 * 1. Fetching training batches from DataIngestion loader.
 * 2. Splitting input tokens and target labels.
 * 3. Forward, Loss, Backward, and weight optimization update steps.
 * 4. Scheduling learning rate using cosine decay with warmup.
 */

#pragma once

#include "Transformer.hpp"
#include "Optimizer.hpp"
#include "DataIngestion.hpp"
#include "Positional.hpp"
#include "Loss.hpp"
#include <cstddef>

/**
 * @brief Configuration parameters for Trainer.
 */
struct TrainerConfig {
  size_t max_steps = 100;      ///< Maximum training steps.
  size_t warmup_steps = 10;    ///< Warmup steps for learning rate scheduling.
  float lr_max = 3e-4f;        ///< Peak learning rate.
  float lr_min = 3e-5f;        ///< Minimum learning rate after decay.
  size_t log_interval = 10;    ///< Step interval for output logs.

  std::string checkpoint_dir = "checkpoints";
  std::string metrics_filepath = "metrics.csv";
  size_t checkpoint_interval = 500;
  size_t keep_last_n_checkpoints = 3;
  bool resume = true;

  /// Path to a specific checkpoint to load at startup (e.g. a Python-converted
  /// model). When empty, the latest checkpoint in checkpoint_dir is auto-loaded.
  std::string init_checkpoint = "";
};

/**
 * @brief Class orchestrating the model pre-training iteration loops.
 */
class Trainer {
public:
  /**
   * @brief Construct a new Trainer object.
   *
   * @param config Configurations parameters.
   * @param model Reference to the Transformer model.
   * @param optimizer Reference to the parameter optimizer.
   * @param data_loader Reference to the data loader ingestion stream.
   * @param rope Reference to the pre-allocated RoPE positional rotations table.
   */
  Trainer(const TrainerConfig &config, Transformer &model, Optimizer &optimizer,
          DataIngestion &data_loader, const RoPE &rope);

  /**
   * @brief Execute the complete training loop.
   */
  void train();

  /**
   * @brief Computes learning rate schedule for the current step.
   *
   * @param step Current training optimization step.
   * @return float Scheduled learning rate.
   */
  float get_scheduled_lr(size_t step) const;

private:
  TrainerConfig config_;
  Transformer &model_;
  Optimizer &optimizer_;
  DataIngestion &data_loader_;
  const RoPE &rope_;
  // Persistent loss object: grad_logits_ buffer (13 GB) is reused across steps
  // instead of being re-allocated every step.
  CrossEntropyLoss lf_;

  void _truncate_metrics_file(size_t step_cutoff);
};
