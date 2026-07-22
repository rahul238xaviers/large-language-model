/**
 * @file Trainer.cpp
 * @brief Implementation of the Trainer class orchestrating model pre-training
 */

#include "Trainer.hpp"
#include "Loss.hpp"
#include "Checkpoint.hpp"
#include "gpu_kernel/MetalBridge.hpp"
#include "gemm_profiler.hpp"
#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/resource.h>

/**
 * @brief Construct a new Trainer object.
 *
 * @param config TrainerConfig parameters.
 * @param model Transformer model to train.
 * @param optimizer Optimizer to update parameters.
 * @param data_loader DataIngestion loader.
 * @param rope Rotary position embedding helper.
 */
Trainer::Trainer(const TrainerConfig &config, Transformer &model, Optimizer &optimizer,
                 DataIngestion &data_loader, const RoPE &rope)
    : config_(config), model_(model), optimizer_(optimizer),
      data_loader_(data_loader), rope_(rope) {}

/**
 * @brief Computes learning rate based on linear warmup and cosine decay.
 *
 * @param step Active training step.
 * @return float Learning rate.
 */
float Trainer::get_scheduled_lr(size_t step) const {
  if (step < config_.warmup_steps) {
    return config_.lr_max * static_cast<float>(step) / static_cast<float>(config_.warmup_steps);
  }
  if (step >= config_.max_steps) {
    return config_.lr_min;
  }
  float ratio = static_cast<float>(step - config_.warmup_steps) /
                static_cast<float>(config_.max_steps - config_.warmup_steps);
  float cos_decay = 0.5f * (1.0f + std::cos(3.1415926535f * ratio));
  return config_.lr_min + cos_decay * (config_.lr_max - config_.lr_min);
}

/**
 * @brief Maps BPE data loader lists into parallel token and target prediction matrices.
 */
static void prepare_batch_tensors(
    const std::vector<std::vector<int>> &batch, size_t batch_size, size_t seq_len,
    Tensor &tokens, Tensor &targets) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len - 1; ++s) {
      tokens(b, s) = static_cast<float>(batch[b][s]);
      targets(b, s) = static_cast<float>(batch[b][s + 1]);
    }
  }
}

/**
 * @brief Performs forward, loss, backward, and parameter optimization updates for one step.
 */
static void run_training_step(
    Transformer &model, Optimizer &optimizer, const RoPE &rope,
    const Tensor &tokens, const Tensor &targets, float lr,
    std::vector<Tensor> &grad_w_gate, std::vector<Tensor> &grad_w_up,
    std::vector<Tensor> &grad_w_down, std::vector<Tensor> &grad_Wq,
    std::vector<Tensor> &grad_Wk, std::vector<Tensor> &grad_Wv,
    std::vector<Tensor> &grad_Wo, Tensor &grad_embeddings,
    Tensor &grad_output_projection, float &loss_out) {

  auto t0 = std::chrono::high_resolution_clock::now();
  Tensor logits = model.forward(tokens);
  auto t1 = std::chrono::high_resolution_clock::now();
  
  CrossEntropyLoss loss_fn;
  loss_out = loss_fn.forward(logits, targets);  // GPU encodes but doesn't execute yet
  auto t2 = std::chrono::high_resolution_clock::now();
  
  // IMPORTANT: Do NOT call loss_fn.backward() — that would COPY grad_logits_,
  // creating a new Tensor with a different data pointer.  The GPU kernels
  // inside model.backward() must read from the SAME buffer that cross_entropy
  // wrote to.  Pass the loss function's internal gradient tensor directly.
  const Tensor &grad_logits = loss_fn.grad_logits();
  auto t3 = std::chrono::high_resolution_clock::now();

  grad_embeddings.fill(0.0f);
  grad_output_projection.fill(0.0f);
  for (size_t l = 0; l < model.layers().size(); ++l) {
    grad_w_gate[l].fill(0.0f);
    grad_w_up[l].fill(0.0f);
    grad_w_down[l].fill(0.0f);
    grad_Wq[l].fill(0.0f);
    grad_Wk[l].fill(0.0f);
    grad_Wv[l].fill(0.0f);
    grad_Wo[l].fill(0.0f);
  }

  model.backward(grad_logits, tokens, grad_w_gate, grad_w_up, grad_w_down,
                 grad_Wq, grad_Wk, grad_Wv, grad_Wo, grad_embeddings,
                 grad_output_projection, rope);
  auto t4 = std::chrono::high_resolution_clock::now();

  optimizer.step(lr);
  auto t5 = std::chrono::high_resolution_clock::now();

  double ms_fwd      = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
  double ms_loss_fwd = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
  double ms_loss_bwd = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;
  double ms_bwd      = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0;
  double ms_opt      = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() / 1000.0;

  std::cout << "[PROFILE] Model Forward (queue):  " << ms_fwd      << " ms" << std::endl;
  std::cout << "[PROFILE] Loss Forward (queue):   " << ms_loss_fwd << " ms" << std::endl;
  std::cout << "[PROFILE] Loss Backward (queue):  " << ms_loss_bwd << " ms" << std::endl;
  std::cout << "[PROFILE] Model Backward (queue): " << ms_bwd      << " ms" << std::endl;
  std::cout << "[PROFILE] Optimizer Step (queue): " << ms_opt      << " ms" << std::endl;
}

/**
 * @brief Executes the pre-training loop for the configured number of steps.
 *
 * Each step gets a token batch from data loader, computes learning rate,
 * runs the training step (forward, loss, backward), and logs stats.
 */
namespace fs = std::filesystem;

void Trainer::_truncate_metrics_file(size_t step_cutoff) {
  if (!fs::exists(config_.metrics_filepath)) return;
  
  std::vector<std::string> lines;
  std::ifstream file(config_.metrics_filepath);
  if (file.is_open()) {
    std::string line;
    std::string header;
    if (std::getline(file, header)) {
      lines.push_back(header);
    }
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string step_str;
      if (std::getline(ss, step_str, ',')) {
        try {
          size_t step_val = std::stoull(step_str);
          if (step_val < step_cutoff) {
            lines.push_back(line);
          }
        } catch (...) {}
      }
    }
    file.close();
  }

  std::ofstream out(config_.metrics_filepath);
  if (out.is_open()) {
    for (const auto &line : lines) {
      out << line << "\n";
    }
    out.close();
    std::cout << "[INFO] Truncated metrics file to step " << step_cutoff - 1 << std::endl;
  }
}

/**
 * @brief Executes the pre-training loop for the configured number of steps.
 *
 * Each step gets a token batch from data loader, computes learning rate,
 * runs the training step (forward, loss, backward), and logs stats.
 */
void Trainer::train() {
  size_t batch_size = data_loader_.batch_size();
  
  std::vector<Tensor> grad_w_gate, grad_w_up, grad_w_down, grad_Wq, grad_Wk, grad_Wv, grad_Wo;
  for (const auto &layer : model_.layers()) {
    grad_w_gate.push_back(Tensor(layer.w_gate.shape(), 0.0f));
    grad_w_up.push_back(Tensor(layer.w_up.shape(), 0.0f));
    grad_w_down.push_back(Tensor(layer.w_down.shape(), 0.0f));
    grad_Wq.push_back(Tensor(layer.attn.Wq().shape(), 0.0f));
    grad_Wk.push_back(Tensor(layer.attn.Wk().shape(), 0.0f));
    grad_Wv.push_back(Tensor(layer.attn.Wv().shape(), 0.0f));
    grad_Wo.push_back(Tensor(layer.attn.Wo().shape(), 0.0f));
  }
  Tensor grad_embeddings(model_.token_embeddings().shape(), 0.0f);
  Tensor grad_output_projection(model_.output_projection().shape(), 0.0f);

  // Register parameters inside optimizer
  optimizer_.register_parameter(&model_.token_embeddings(), &grad_embeddings);
  optimizer_.register_parameter(&model_.output_projection(), &grad_output_projection);
  for (size_t l = 0; l < model_.layers().size(); ++l) {
    optimizer_.register_parameter(&model_.layers()[l].w_gate, &grad_w_gate[l]);
    optimizer_.register_parameter(&model_.layers()[l].w_up, &grad_w_up[l]);
    optimizer_.register_parameter(&model_.layers()[l].w_down, &grad_w_down[l]);
    optimizer_.register_parameter(&model_.layers()[l].attn.Wq(), &grad_Wq[l]);
    optimizer_.register_parameter(&model_.layers()[l].attn.Wk(), &grad_Wk[l]);
    optimizer_.register_parameter(&model_.layers()[l].attn.Wv(), &grad_Wv[l]);
    optimizer_.register_parameter(&model_.layers()[l].attn.Wo(), &grad_Wo[l]);
  }

  // 1. Checkpoint auto-resume
  size_t start_step = 0;
  std::string latest_ckpt = "";
  if (config_.resume && fs::exists(config_.checkpoint_dir)) {
    size_t max_step_num = 0;
    for (const auto &entry : fs::directory_iterator(config_.checkpoint_dir)) {
      if (entry.is_regular_file()) {
        std::string filename = entry.path().filename().string();
        if (filename.rfind("step_", 0) == 0 &&
            filename.find(".safetensors") != std::string::npos &&
            filename.find(".opt.safetensors") == std::string::npos) {
          size_t start = 5;
          size_t end = filename.find(".safetensors");
          if (end != std::string::npos && end > start) {
            try {
              size_t step_num = std::stoull(filename.substr(start, end - start));
              if (step_num > max_step_num) {
                max_step_num = step_num;
                latest_ckpt = entry.path().string();
              }
            } catch (...) {}
          }
        }
      }
    }
    if (max_step_num > 0) {
      std::cout << "[INFO] Auto-discovered latest checkpoint: " << latest_ckpt 
                << ". Resuming weights and starting from step " << max_step_num << std::endl;
      if (Checkpoint::load(latest_ckpt, model_, optimizer_, start_step)) {
        std::cout << "[INFO] Resumed training state successfully. start_step = " << start_step << std::endl;
        _truncate_metrics_file(start_step);
      } else {
        std::cerr << "[WARNING] Failed to load checkpoint. Starting training from scratch." << std::endl;
        start_step = 0;
      }
    }
  }

  // 2. Token skipping for already processed tokens
  if (start_step > 0) {
    size_t num_sequences_to_skip = start_step * batch_size;
    data_loader_.skip_sequences(num_sequences_to_skip);
  }

  // 3. Initialize metrics.csv
  bool metrics_existed = fs::exists(config_.metrics_filepath);
  if (start_step == 0 || !metrics_existed) {
    std::ofstream out(config_.metrics_filepath);
    if (out.is_open()) {
      out << "step,train_loss,tokens_per_sec,learning_rate,vram_usage_gb,mfu_pct,step_time_ms,gpu_calls,gpu_time_ms,cpu_calls,cpu_time_ms,gpu_active_pct\n";
      out.close();
    }
  }

  std::cout << "[INFO] Commencing pre-training loop..." << std::endl;

  for (size_t step = start_step; step < config_.max_steps; ++step) {
    std::cout << "[DEBUG] Starting step: " << step << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<int>> batch = data_loader_.get_batch();
    if (batch.empty()) {
      std::cout << "[INFO] End of token stream reached. Stopping training." << std::endl;
      break;
    }

    size_t active_batch_size = batch.size();
    size_t seq_len = batch[0].size();
    
    Tensor tokens({active_batch_size, seq_len - 1});
    Tensor targets({active_batch_size, seq_len - 1});
    prepare_batch_tensors(batch, active_batch_size, seq_len, tokens, targets);

    float lr = get_scheduled_lr(step);
    float loss = 0.0f;

    // ── Single command buffer for the ENTIRE step ─────────────────────
    // Forward, loss, backward, and optimizer all encode into one buffer.
    // GPU executes continuously without waiting for CPU between phases.
    // CPU cannot read the loss value until end_scope() completes.
    metal_bridge::initialize();
    metal_bridge::reconcile_buffers();

    metal_bridge::begin_scope();

    run_training_step(model_, optimizer_, rope_, tokens, targets, lr,
                      grad_w_gate, grad_w_up, grad_w_down, grad_Wq,
                      grad_Wk, grad_Wv, grad_Wo, grad_embeddings,
                      grad_output_projection, loss);

    // loss inside run_training_step is from loss_fn.forward() which is
    // garbage (GPU hasn't executed).  Overwrite after commit.

    metal_bridge::end_scope();

    // ── GPU has finished — read the real loss ─────────────────────────
    loss = metal_bridge::get_last_loss();

    auto end_time = std::chrono::high_resolution_clock::now();
    double step_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

    double tokens_per_sec = (active_batch_size * seq_len) / (step_ms / 1000.0);
    struct rusage usage;
    double vram_gb = 0.0;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      vram_gb = static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
    }
    const auto& mcfg = model_.config();
    size_t param_count = mcfg.vocab_size * mcfg.hidden_dim +
        mcfg.n_layers * (
            mcfg.hidden_dim * (mcfg.n_heads * mcfg.head_dim) +
            mcfg.hidden_dim * (mcfg.n_kv_heads * mcfg.head_dim) * 2 +
            (mcfg.n_heads * mcfg.head_dim) * mcfg.hidden_dim +
            mcfg.hidden_dim * mcfg.intermediate_dim * 2 +
            mcfg.intermediate_dim * mcfg.hidden_dim +
            mcfg.hidden_dim * 2
        ) + mcfg.hidden_dim + mcfg.hidden_dim * mcfg.vocab_size;
    double mfu_pct = ((6.0 * param_count * (active_batch_size * seq_len)) / (step_ms / 1000.0)) / (28.3e12) * 100.0;
    double gpu_active_pct = (metal_bridge::accum_gpu_time_ms / step_ms) * 100.0;

    // 4. Log to metrics.csv
    std::ofstream out(config_.metrics_filepath, std::ios::app);
    if (out.is_open()) {
      out << step << ","
          << loss << ","
          << tokens_per_sec << ","
          << lr << ","
          << vram_gb << ","
          << mfu_pct << ","
          << step_ms << ","
          << metal_bridge::count_gpu_calls << ","
          << metal_bridge::accum_gpu_time_ms << ","
          << metal_bridge::count_cpu_calls << ","
          << metal_bridge::accum_cpu_time_ms << ","
          << gpu_active_pct << "\n";
      out.close();
    }

    if (step % config_.log_interval == 0 || step == config_.max_steps - 1) {
      std::cout << "  Step " << std::setw(4) << step 
                << " | Loss: " << std::fixed << std::setprecision(5) << loss
                << " | LR: " << std::scientific << std::setprecision(4) << lr
                << " | Latency: " << std::fixed << std::setprecision(2) << step_ms << " ms"
                << " | MFU: " << std::fixed << std::setprecision(2) << mfu_pct << "%"
                << " | GPU Active: " << std::fixed << std::setprecision(2) << gpu_active_pct << "%"
                << " [Calls: " << metal_bridge::count_gpu_calls << " GPU / " << metal_bridge::count_cpu_calls << " CPU]"
                << std::endl;
    }
    metal_bridge::reset_profile_stats();

    // 5. Save checkpoints
    if ((step + 1) % config_.checkpoint_interval == 0 || (step + 1) == config_.max_steps) {
      fs::create_directories(config_.checkpoint_dir);
      char ckpt_filename[256];
      std::snprintf(ckpt_filename, sizeof(ckpt_filename), "step_%07d.safetensors", static_cast<int>(step + 1));
      std::string ckpt_path = (fs::path(config_.checkpoint_dir) / ckpt_filename).string();
      
      std::cout << "[INFO] Saving checkpoint to " << ckpt_path << "..." << std::endl;
      if (Checkpoint::save(ckpt_path, model_, optimizer_, step + 1)) {
        std::cout << "[INFO] Checkpoint saved successfully: " << ckpt_filename << std::endl;
        
        // 6. Prune old checkpoints
        if (config_.keep_last_n_checkpoints > 0) {
          std::vector<std::pair<size_t, std::string>> ckpt_files;
          for (const auto &entry : fs::directory_iterator(config_.checkpoint_dir)) {
            if (entry.is_regular_file()) {
              std::string fn = entry.path().filename().string();
              if (fn.rfind("step_", 0) == 0 &&
                  fn.find(".safetensors") != std::string::npos &&
                  fn.find(".opt.safetensors") == std::string::npos) {
                size_t start = 5;
                size_t end = fn.find(".safetensors");
                try {
                  size_t step_num = std::stoull(fn.substr(start, end - start));
                  ckpt_files.push_back({step_num, entry.path().string()});
                } catch (...) {}
              }
            }
          }
          if (ckpt_files.size() > config_.keep_last_n_checkpoints) {
            std::sort(ckpt_files.begin(), ckpt_files.end());
            size_t to_remove = ckpt_files.size() - config_.keep_last_n_checkpoints;
            for (size_t i = 0; i < to_remove; ++i) {
              std::string path_to_remove = ckpt_files[i].second;
              fs::remove(path_to_remove);
              
              std::string opt_path = path_to_remove;
              size_t pos = opt_path.find(".safetensors");
              if (pos != std::string::npos) {
                opt_path.replace(pos, 12, ".opt.safetensors");
              }
              fs::remove(opt_path);
              std::cout << "[INFO] Pruned old checkpoint: " << path_to_remove << std::endl;
            }
          }
        }
      } else {
        std::cerr << "[ERROR] Failed to save checkpoint: " << ckpt_filename << std::endl;
      }
    }
  }
}
