#ifndef DATA_INGESTION_HPP
#define DATA_INGESTION_HPP

#include <arrow/api.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class DataIngestion {
public:
  DataIngestion(const std::string &data_dir,
                size_t max_shard_bytes = 100ULL << 20);
  std::vector<std::vector<int>> get_batch();

private:
  std::string data_dir_;
  std::vector<std::shared_ptr<arrow::Table>> shards_;
  std::vector<std::string> file_paths_;
  std::vector<std::vector<int>> token_batches_;
  size_t max_shard_bytes_;
  int current_shard_idx_ = 0;
  int current_batch_idx_ = 0;
  std::unordered_map<std::string, int> vocab_;
  bool load_parquet_shard(const std::string &file_path);
  void tokenize_cpp_corpus(std::shared_ptr<arrow::Table> table);
  void generate_training_sequences();
  static void bpe_encode(const std::string &text, std::vector<int> &tokens,
                         const std::unordered_map<std::string, int> &vocab);
};

#endif // DATA_INGESTION_HPP