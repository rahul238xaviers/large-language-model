#include "DataIngestion.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

DataIngestion::DataIngestion(const std::string &data_dir,
                             const std::string &single_data_file, size_t max_shard_bytes,
                             size_t batch_size, size_t sequence_length,
                             const std::string &vocab_path)
    : batch_size_(batch_size), sequence_length_(sequence_length),
      current_batch_idx_(0) {
  
  std::string bin_path = "data/datasets/rust/train.bin";
  if (getenv("VERBOSE")) std::cout << "[INFO] Loading binary tokens from " << bin_path << "..." << std::endl;
  
  std::ifstream file(bin_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open " << bin_path << ". Run Python pretokenization first!" << std::endl;
    exit(1);
  }
  
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  size_t num_tokens = size / sizeof(uint32_t);
  if (getenv("VERBOSE")) std::cout << "[INFO] Found " << num_tokens << " tokens in binary." << std::endl;
  
  std::vector<uint32_t> raw_tokens(num_tokens);
  if (file.read(reinterpret_cast<char*>(raw_tokens.data()), size)) {
    flat_tokens_.assign(raw_tokens.begin(), raw_tokens.end());
  } else {
    std::cerr << "Error: Failed to read tokens from " << bin_path << std::endl;
    exit(1);
  }
  
  generate_training_sequences();
}

DataIngestion::~DataIngestion() = default;

void DataIngestion::generate_training_sequences() {
  std::cout << "[INFO] Generating training sequences..." << std::endl;
  token_batches_.clear();
  current_batch_idx_ = 0;

  size_t step = sequence_length_ + 1;
  if (flat_tokens_.size() < step) {
    return;
  }

  for (size_t i = 0; i + step <= flat_tokens_.size(); i += step) {
    std::vector<int> sequence(flat_tokens_.begin() + i,
                              flat_tokens_.begin() + i + step);
    token_batches_.push_back(sequence);
  }
  if (getenv("VERBOSE")) std::cout << "[INFO] Generated " << token_batches_.size() << " training sequences." << std::endl;
}

std::vector<std::vector<int>> DataIngestion::get_batch() {
  std::vector<std::vector<int>> batch;
  batch.reserve(batch_size_);

  if (token_batches_.empty()) {
    return {};
  }

  for (size_t i = 0; i < batch_size_; ++i) {
    if (current_batch_idx_ >= token_batches_.size()) {
      current_batch_idx_ = 0; // Wrap around to the start of the token stream
    }
    batch.push_back(token_batches_[current_batch_idx_++]);
  }
  return batch;
}

void DataIngestion::skip_sequences(size_t num_sequences) {
  if (token_batches_.empty()) return;
  current_batch_idx_ = (current_batch_idx_ + num_sequences) % token_batches_.size();
  if (getenv("VERBOSE")) {
    std::cout << "[INFO] Resume | Skipped " << num_sequences << " already-processed sequences (" 
              << num_sequences * (sequence_length_ + 1) << " tokens) from data stream." << std::endl;
  }
}
