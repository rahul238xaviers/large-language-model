/**
 * @file DataIngestion.hpp
 * @brief Data Ingestion pipeline declaration
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>

class DataIngestion {
public:
  DataIngestion(const std::string &data_dir,
                const std::string &single_data_file, size_t max_shard_bytes,
                size_t batch_size, size_t sequence_length,
                const std::string &vocab_path);
  ~DataIngestion();
  std::vector<std::vector<int>> get_batch();
  void skip_sequences(size_t num_sequences);
  size_t batch_size() const { return batch_size_; }
  static constexpr int EOT = 100257;

private:
  size_t batch_size_;
  size_t sequence_length_;
  std::vector<int> flat_tokens_;
  std::vector<std::vector<int>> token_batches_;
  size_t current_batch_idx_ = 0;
  
  void generate_training_sequences();
};