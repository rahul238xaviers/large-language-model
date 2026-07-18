/**
 * @file RunTrainer.cpp
 * @brief Command-line utility to run production pre-training for the 1.6B GPT model in C++
 */

#include "Trainer.hpp"
#include "Checkpoint.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

// WHAT: Helper function to initialize GPT-2/LLaMA style weights using standard normal distributions.
// WHY: Ensures that if we train from scratch (no checkpoint), the network has standard, stable weight distributions.
static void initialize_gpt_weights(Transformer &model) {
  std::mt19937 gen(42);
  
  size_t hidden_dim = model.token_embeddings().shape()[1];
  float stddev = 1.0f / std::sqrt(static_cast<float>(hidden_dim));
  std::normal_distribution<float> dist(0.0f, stddev);
  std::normal_distribution<float> emb_dist(0.0f, 0.02f);

  // Initialize embeddings
  for (auto &val : model.token_embeddings().data()) {
    val = emb_dist(gen);
  }
  // Initialize output projection
  for (auto &val : model.output_projection().data()) {
    val = dist(gen);
  }
  // Initialize RMSNorm scales to 1.0
  model.final_norm().weight().fill(1.0f);

  for (auto &layer : model.layers()) {
    layer.attn_norm.weight().fill(1.0f);
    layer.ffn_norm.weight().fill(1.0f);

    for (auto &val : layer.w_gate.data()) val = dist(gen);
    for (auto &val : layer.w_up.data()) val = dist(gen);
    for (auto &val : layer.w_down.data()) val = dist(gen);

    for (auto &val : layer.attn.Wq().data()) val = dist(gen);
    for (auto &val : layer.attn.Wk().data()) val = dist(gen);
    for (auto &val : layer.attn.Wv().data()) val = dist(gen);
    for (auto &val : layer.attn.Wo().data()) val = dist(gen);
  }
}

int main(int argc, char* argv[]) {
  bool use_gpu = true; // Enabled by default for production pre-training
  size_t batch_size = 4;
  size_t max_steps = 10000;
  std::string data_dir = "data/raw_chunks/bigcode/the-stack-v2/data/c++/";
  std::string vocab_path = "data/raw_chunks/vocabulary/cl100k_base.tiktoken";
  std::string checkpoint_dir = "checkpoints";
  std::string metrics_filepath = "metrics.csv";
  bool resume = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--gpu") {
      use_gpu = true;
    } else if (arg == "--cpu") {
      use_gpu = false;
    } else if (arg == "--batch_size" && i + 1 < argc) {
      batch_size = std::stoull(argv[++i]);
    } else if (arg == "--max_steps" && i + 1 < argc) {
      max_steps = std::stoull(argv[++i]);
    } else if (arg == "--data_dir" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (arg == "--vocab_path" && i + 1 < argc) {
      vocab_path = argv[++i];
    } else if (arg == "--checkpoint_dir" && i + 1 < argc) {
      checkpoint_dir = argv[++i];
    } else if (arg == "--metrics" && i + 1 < argc) {
      metrics_filepath = argv[++i];
    } else if (arg == "--no-resume") {
      resume = false;
    }
  }

  // Set GPU execution environment flag
  if (use_gpu) {
    setenv("GPU_ENABLED", "1", 1);
    std::cout << "[INFO] GPU execution enabled (GPU_ENABLED=1)" << std::endl;
  } else {
    setenv("GPU_ENABLED", "0", 1);
    std::cout << "[INFO] CPU execution enabled (GPU_ENABLED=0)" << std::endl;
  }

  // 1.6B parameter Rust-GPT production configurations
  ModelConfig config;
  config.vocab_size = 100277; // cl100k_base
  config.hidden_dim = 2048;
  config.intermediate_dim = 5461;
  config.n_layers = 24;
  config.n_heads = 16;
  config.n_kv_heads = 8;
  config.head_dim = 128;
  config.max_seq_len = 2048;
  config.rope_base = 10000.0f;
  config.rms_norm_eps = 1e-5f;

  std::cout << "==========================================================" << std::endl;
  std::cout << "    RUST-GPT 1.6B LLM C++ PRODUCTION PRE-TRAINING ENGINE" << std::endl;
  std::cout << "==========================================================" << std::endl;
  std::cout << "  n_layer:            " << config.n_layers << std::endl;
  std::cout << "  n_embd (hidden):    " << config.hidden_dim << std::endl;
  std::cout << "  n_head:             " << config.n_heads << std::endl;
  std::cout << "  n_kv_heads (GQA):   " << config.n_kv_heads << std::endl;
  std::cout << "  block_size:         " << config.max_seq_len << std::endl;
  std::cout << "  vocab_size:         " << config.vocab_size << std::endl;
  std::cout << "  batch_size:         " << batch_size << std::endl;
  std::cout << "  max_steps:          " << max_steps << std::endl;
  std::cout << "  checkpoint_dir:     " << checkpoint_dir << std::endl;
  std::cout << "  metrics_file:       " << metrics_filepath << std::endl;
  std::cout << "  auto_resume:        " << (resume ? "true" : "false") << std::endl;
  std::cout << "==========================================================" << std::endl;

  std::cout << "[INFO] Initialising model memory allocation..." << std::endl;
  Transformer model(config);

  std::cout << "[INFO] Initialising weight parameters (Scratch initialization)..." << std::endl;
  initialize_gpt_weights(model);

  std::cout << "[INFO] Initialising RoPE rotations..." << std::endl;
  RoPE rope(config.head_dim, config.max_seq_len, config.rope_base);

  std::cout << "[INFO] Initialising AdamW Optimizer (decay=0.1)..." << std::endl;
  AdamWOptimizer optimizer(0.9f, 0.999f, 1e-8f, 0.1f); 

  std::cout << "[INFO] Initialising Parquet Data Ingestion..." << std::endl;
  DataIngestion data_loader(data_dir, "", 500 * 1024 * 1024, batch_size, config.max_seq_len, vocab_path);

  TrainerConfig train_cfg;
  train_cfg.max_steps = max_steps;
  train_cfg.warmup_steps = 500;
  train_cfg.lr_max = 3e-4f;
  train_cfg.lr_min = 3e-5f;
  train_cfg.log_interval = 10;
  train_cfg.checkpoint_interval = 500;
  train_cfg.keep_last_n_checkpoints = 3;
  train_cfg.checkpoint_dir = checkpoint_dir;
  train_cfg.metrics_filepath = metrics_filepath;
  train_cfg.resume = resume;

  Trainer trainer(train_cfg, model, optimizer, data_loader, rope);

  std::cout << "[INFO] Starting training pipeline execution..." << std::endl;
  trainer.train();

  std::cout << "[INFO] Pre-training loop completed successfully." << std::endl;
  return 0;
}
