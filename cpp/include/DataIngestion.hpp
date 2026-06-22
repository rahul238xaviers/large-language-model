#ifndef DATA_INGESTION_HPP
#define DATA_INGESTION_HPP

#include <arrow/api.h>
#include <arrow/status.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class DataIngestion {
public:
  DataIngestion(const std::string &data_dir,
                const std::string &single_data_file, size_t max_shard_bytes,
                size_t batch_size, size_t sequence_length,
                const std::string &vocab_path);
  std::vector<std::vector<int>> get_batch();
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

  struct Part {
    size_t start;
    size_t length;
  };
  std::vector<Part> part_buffer_;

  std::unordered_map<std::string, int> vocab_;
  arrow::Status load_parquet_shard(const std::string &file_path);
  void tokenize_cpp_corpus(std::shared_ptr<arrow::Table> table,
                           const std::string col);
  void generate_training_sequences();
  void bpe_encode(const std::string &text, std::vector<int> &tokens,
                  const std::unordered_map<std::string, int> &vocab);
  std::string base64_decode(const std::string &encoded_text);
  bool load_vocabulary(const std::string &vocab_path);
};

#endif // DATA_INGESTION_HPP