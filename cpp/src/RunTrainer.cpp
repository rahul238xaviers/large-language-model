/**
 * @file RunTrainer.cpp
 * @brief Command-line utility to run production pre-training for the 1.6B GPT model in C++
 */

#include "Trainer.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

// Custom stream buffer to duplicate stdout/stderr to a log file
class TeeBuffer : public std::streambuf {
public:
  TeeBuffer(std::streambuf * sb1, std::streambuf * sb2) : sb1(sb1), sb2(sb2) {}
protected:
  virtual int overflow(int c) override {
    if (c == EOF) return !EOF;
    int r1 = sb1->sputc(c);
    int r2 = sb2->sputc(c);
    return (r1 == EOF || r2 == EOF) ? EOF : c;
  }
  virtual int sync() override {
    int r1 = sb1->pubsync();
    int r2 = sb2->pubsync();
    return (r1 == 0 && r2 == 0) ? 0 : -1;
  }
private:
  std::streambuf * sb1;
  std::streambuf * sb2;
};

static std::string get_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
  return ss.str();
}

// Format floats like Python's json.dumps (integral values get a trailing ".0").
static std::string fmt_float(float v) {
  std::ostringstream ss;
  ss << v;
  std::string s = ss.str();
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
      s.find("inf") == std::string::npos && s.find("nan") == std::string::npos) {
    s += ".0";
  }
  return s;
}

static void save_config_json(const std::string &filepath, const ModelConfig &config,
                             size_t batch_size, size_t max_steps,
                             const TrainerConfig &train_cfg,
                             float opt_beta1, float opt_beta2, float opt_eps,
                             float opt_weight_decay, float grad_clip_norm) {
  // CPP does not implement gradient accumulation yet; one micro-batch == one
  // optimizer step.  Effective batch / tokens mirror the Python config schema.
  const size_t gradient_accumulation_steps = 1;
  const size_t effective_batch_size = batch_size * gradient_accumulation_steps;

  std::ofstream out(filepath);
  if (out.is_open()) {
    out << "{\n";
    out << "    \"n_layer\": " << config.n_layers << ",\n";
    out << "    \"n_embd\": " << config.hidden_dim << ",\n";
    out << "    \"n_head\": " << config.n_heads << ",\n";
    out << "    \"n_kv_head\": " << config.n_kv_heads << ",\n";
    out << "    \"head_dim\": " << config.head_dim << ",\n";
    out << "    \"block_size\": " << config.max_seq_len << ",\n";
    out << "    \"vocab_size\": " << config.vocab_size << ",\n";
    out << "    \"micro_batch_size\": " << batch_size << ",\n";
    out << "    \"gradient_accumulation_steps\": " << gradient_accumulation_steps << ",\n";
    out << "    \"learning_rate\": " << fmt_float(train_cfg.lr_max) << ",\n";
    out << "    \"min_lr\": " << fmt_float(train_cfg.lr_min) << ",\n";
    out << "    \"warmup_iters\": " << train_cfg.warmup_steps << ",\n";
    out << "    \"max_iters\": " << max_steps << ",\n";
    out << "    \"weight_decay\": " << fmt_float(opt_weight_decay) << ",\n";
    out << "    \"grad_clip_norm\": " << fmt_float(grad_clip_norm) << ",\n";
    out << "    \"beta1\": " << fmt_float(opt_beta1) << ",\n";
    out << "    \"beta2\": " << fmt_float(opt_beta2) << ",\n";
    out << "    \"eps\": " << fmt_float(opt_eps) << ",\n";
    out << "    \"dtype\": \"bfloat16\",\n";
    out << "    \"save_interval\": " << train_cfg.checkpoint_interval << ",\n";
    out << "    \"keep_checkpoints\": " << train_cfg.keep_last_n_checkpoints << ",\n";
    out << "    \"checkpoint_dir\": \"" << train_cfg.checkpoint_dir << "\",\n";
    out << "    \"effective_batch_size\": " << effective_batch_size << ",\n";
    out << "    \"total_tokens_per_iter\": " << effective_batch_size * config.max_seq_len << "\n";
    out << "}\n";
    out.close();
  }
}

// WHAT: Helper function to initialize GPT-2/LLaMA style weights using standard normal distributions.
// WHY: Ensures that if we train from scratch (no checkpoint), the network has standard, stable weight distributions.
static void initialize_gpt_weights(Transformer &model) {
  std::mt19937 gen(42);
  
  size_t hidden_dim = model.token_embeddings().shape()[1];
  float stddev = 1.0f / std::sqrt(static_cast<float>(hidden_dim));
  std::normal_distribution<float> dist(0.0f, stddev);
  std::normal_distribution<float> emb_dist(0.0f, 0.02f);

  // Initialize embeddings (BF16 storage)
  {
    uint16_t* w = (uint16_t*)model.token_embeddings().data_bf16();
    for (size_t _i = 0; _i < model.token_embeddings().num_elements(); ++_i) w[_i] = float_to_bf16((float)emb_dist(gen));
  }
  // Initialize output projection (BF16 storage)
  {
    uint16_t* w = (uint16_t*)model.output_projection().data_bf16();
    for (size_t _i = 0; _i < model.output_projection().num_elements(); ++_i) w[_i] = float_to_bf16((float)dist(gen));
  }
  // Initialize RMSNorm scales to 1.0
  model.final_norm().weight().fill(1.0f);

  for (auto &layer : model.layers()) {
    layer.attn_norm.weight().fill(1.0f);
    layer.ffn_norm.weight().fill(1.0f);

    auto fill_bf16_weight = [](Tensor &t, std::mt19937 &g, std::normal_distribution<float> &d) {
      uint16_t* w = (uint16_t*)t.data_bf16();
      for (size_t i = 0; i < t.num_elements(); ++i) w[i] = float_to_bf16((float)d(g));
    };
    fill_bf16_weight(layer.w_gate, gen, dist);
    fill_bf16_weight(layer.w_up, gen, dist);
    fill_bf16_weight(layer.w_down, gen, dist);
    fill_bf16_weight(layer.attn.Wq(), gen, dist);
    fill_bf16_weight(layer.attn.Wk(), gen, dist);
    fill_bf16_weight(layer.attn.Wv(), gen, dist);
    fill_bf16_weight(layer.attn.Wo(), gen, dist);
  }
}

int main(int argc, char* argv[]) {
  bool use_gpu = true;
  size_t batch_size = 4;
  size_t max_steps = 10000;
  std::string data_dir = "data/datasets/rust/";
  std::string vocab_path = "data/raw_chunks/vocabulary/cl100k_base.tiktoken";
  std::string checkpoint_dir = "cpp/runs";
  std::string metrics_filepath = "metrics.csv";
  std::string init_checkpoint = "";
  bool resume = true;

  // Default 380M parameter Rust-GPT production configurations
  ModelConfig config;
  // Matches Python checkpoint vocab_size (cl100k_base).
  config.vocab_size = 100277; 
  config.hidden_dim = 1024;
  config.intermediate_dim = 2730; // Match Python checkpoint
  config.n_layers = 24;
  config.n_heads = 16;
  config.n_kv_heads = 8;
  config.head_dim = 64;
  config.max_seq_len = 1024;
  config.rope_base = 10000.0f;
  config.rms_norm_eps = 1e-5f;

  size_t checkpoint_interval = 500;

  // AdamW / training hyperparameters (mirror the Python config schema).
  const float opt_beta1 = 0.9f;
  const float opt_beta2 = 0.999f;  // Python used 0.95; CPP uses 0.999.
  const float opt_eps = 1e-8f;
  const float opt_weight_decay = 0.1f;
  const float grad_clip_norm = 1.0f;  // recorded for parity; clipping not enforced

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
    } else if (arg == "--init_checkpoint" && i + 1 < argc) {
      init_checkpoint = argv[++i];
    } else if (arg == "--no-resume") {
      resume = false;
    } else if (arg == "--hidden_dim" && i + 1 < argc) {
      config.hidden_dim = std::stoull(argv[++i]);
    } else if (arg == "--intermediate_dim" && i + 1 < argc) {
      config.intermediate_dim = std::stoull(argv[++i]);
    } else if (arg == "--n_layers" && i + 1 < argc) {
      config.n_layers = std::stoull(argv[++i]);
    } else if (arg == "--n_heads" && i + 1 < argc) {
      config.n_heads = std::stoull(argv[++i]);
    } else if (arg == "--n_kv_heads" && i + 1 < argc) {
      config.n_kv_heads = std::stoull(argv[++i]);
    } else if (arg == "--head_dim" && i + 1 < argc) {
      config.head_dim = std::stoull(argv[++i]);
    } else if (arg == "--max_seq_len" && i + 1 < argc) {
      config.max_seq_len = std::stoull(argv[++i]);
    } else if (arg == "--checkpoint_interval" && i + 1 < argc) {
      checkpoint_interval = std::stoull(argv[++i]);
    }
  }

  // Set GPU execution environment flag
  if (use_gpu) {
    setenv("GPU_ENABLED", "1", 1);
  } else {
    setenv("GPU_ENABLED", "0", 1);
  }

  // 1. Resolve run directory layout to match Python runs/run_YYYYMMDD_HHMMSS/
  std::string run_dir = "";
  std::string log_filepath = "";

  bool found_existing_checkpoint = false;
  if (resume) {
    std::filesystem::path cp_path(checkpoint_dir);
    // Check if cp_path exists and has checkpoints directly
    if (std::filesystem::exists(cp_path) && std::filesystem::is_directory(cp_path)) {
      for (const auto &entry : std::filesystem::directory_iterator(cp_path)) {
        if (entry.path().extension() == ".safetensors") {
          found_existing_checkpoint = true;
          break;
        }
      }
    }
    // Check if cp_path/checkpoints has checkpoints
    if (!found_existing_checkpoint && std::filesystem::exists(cp_path / "checkpoints")) {
      for (const auto &entry : std::filesystem::directory_iterator(cp_path / "checkpoints")) {
        if (entry.path().extension() == ".safetensors") {
          found_existing_checkpoint = true;
          checkpoint_dir = (cp_path / "checkpoints").string();
          break;
        }
      }
    }
    
    if (found_existing_checkpoint) {
      std::filesystem::path resolved_cp_path(checkpoint_dir);
      if (resolved_cp_path.filename() == "checkpoints") {
        run_dir = resolved_cp_path.parent_path().string();
      } else {
        run_dir = resolved_cp_path.string();
      }
      metrics_filepath = (std::filesystem::path(run_dir) / "metrics.csv").string();
      log_filepath = (std::filesystem::path(run_dir) / "train.log").string();
    }
  }

  // Training schedule configuration (shared between config.json and the Trainer).
  TrainerConfig train_cfg;
  train_cfg.max_steps = max_steps;
  if (const char *ws = std::getenv("TRAIN_WARMUP_STEPS")) train_cfg.warmup_steps = std::atoi(ws);
  else train_cfg.warmup_steps = 500;
  if (const char *lr = std::getenv("TRAIN_LR_MAX")) train_cfg.lr_max = std::atof(lr);
  else train_cfg.lr_max = 3e-4f;
  if (const char *lrmin = std::getenv("TRAIN_LR_MIN")) train_cfg.lr_min = std::atof(lrmin);
  else train_cfg.lr_min = 3e-5f;
  train_cfg.log_interval = 1;
  train_cfg.checkpoint_interval = checkpoint_interval;
  train_cfg.keep_last_n_checkpoints = 3;
  train_cfg.checkpoint_dir = checkpoint_dir;
  train_cfg.metrics_filepath = metrics_filepath;
  train_cfg.resume = resume;
  train_cfg.init_checkpoint = init_checkpoint;

  // Create a new run directory if we didn't find an existing run to resume
  if (run_dir.empty()) {
    std::string timestamp = get_timestamp();
    run_dir = "cpp/runs/run_" + timestamp;
    checkpoint_dir = run_dir + "/checkpoints";
    metrics_filepath = run_dir + "/metrics.csv";
    log_filepath = run_dir + "/train.log";
    train_cfg.checkpoint_dir = checkpoint_dir;
    train_cfg.metrics_filepath = metrics_filepath;

    std::filesystem::create_directories(checkpoint_dir);
    save_config_json(run_dir + "/config.json", config, batch_size, max_steps, train_cfg,
                     opt_beta1, opt_beta2, opt_eps, opt_weight_decay, grad_clip_norm);
  } else {
    std::filesystem::create_directories(checkpoint_dir);
  }

  // Redirect stdout and stderr to both standard streams and the train.log file
  std::ofstream log_file(log_filepath, std::ios::app);
  TeeBuffer tee_cout(std::cout.rdbuf(), log_file.rdbuf());
  TeeBuffer tee_cerr(std::cerr.rdbuf(), log_file.rdbuf());
  
  std::streambuf *orig_cout_buf = std::cout.rdbuf(&tee_cout);
  std::streambuf *orig_cerr_buf = std::cerr.rdbuf(&tee_cerr);

  if (use_gpu) {
    std::cout << "[INFO] GPU execution enabled (GPU_ENABLED=1)" << std::endl;
  } else {
    std::cout << "[INFO] CPU execution enabled (GPU_ENABLED=0)" << std::endl;
  }

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
  if (!init_checkpoint.empty())
    std::cout << "  init_checkpoint:    " << init_checkpoint << std::endl;
  std::cout << "  auto_resume:        " << (resume ? "true" : "false") << std::endl;
  std::cout << "==========================================================" << std::endl;

  std::cout << "[INFO] Initialising model memory allocation..." << std::endl;
  Transformer model(config);

  std::cout << "[INFO] Initialising weight parameters (Scratch initialization)..." << std::endl;
  initialize_gpt_weights(model);

  std::cout << "[INFO] Initialising RoPE rotations..." << std::endl;
  RoPE rope(config.head_dim, config.max_seq_len, config.rope_base);

  std::cout << "[INFO] Initialising AdamW Optimizer (decay=0.1)..." << std::endl;
  AdamWOptimizer optimizer(opt_beta1, opt_beta2, opt_eps, opt_weight_decay);

  std::cout << "[INFO] Initialising Parquet Data Ingestion..." << std::endl;
  DataIngestion data_loader(data_dir, "", 500 * 1024 * 1024, batch_size, config.max_seq_len, vocab_path);

  Trainer trainer(train_cfg, model, optimizer, data_loader, rope);

  std::cout << "[INFO] Starting training pipeline execution..." << std::endl;
  trainer.train();

  std::cout << "[INFO] Pre-training loop completed successfully." << std::endl;

  // Restore original stream buffers
  std::cout.rdbuf(orig_cout_buf);
  std::cerr.rdbuf(orig_cerr_buf);
  return 0;
}
