/**
 * @file DataIngestion.hpp
 * @brief Data Ingestion pipeline declaration
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Declares the DataIngestion class which manages loading raw files, tokenization,
 * and streaming batch preparation for training.
 */

#pragma once

#include <arrow/api.h>
#include <arrow/status.h>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

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
  std::string data_dir_;
  std::string single_data_file;
  std::string vocab_path_;
  std::vector<std::shared_ptr<arrow::Table>> shards_;
  std::vector<std::string> file_paths_;
  std::vector<std::vector<int>> token_batches_;
  std::vector<int> flat_tokens_;
  size_t max_shard_bytes_;
  size_t batch_size_;
  size_t sequence_length_;
  int current_shard_idx_ = 0;
  int current_batch_idx_ = 0;

  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  arrow::Status load_parquet_shard(const std::string &file_path);
  void tokenize_cpp_corpus(std::shared_ptr<arrow::Table> table,
                           const std::string col);
  void generate_training_sequences();
  void bpe_encode(const std::string &text, std::vector<int> &tokens);
};