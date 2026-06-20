#include "DataIngestion.hpp"
#include <iostream>

DataIngestion::DataIngestion(const std::string &data_dir,
                             size_t max_shard_bytes) {
  data_dir_ = data_dir;
  max_shard_bytes_ = max_shard_bytes;
}
bool DataIngestion::load_parquet_shard(const std::string &file_path) {
  std::cout << "Loading shard: " << file_path << std::endl;
  return true;
}
void DataIngestion::tokenize_cpp_corpus(std::shared_ptr<arrow::Table> table) {
  std::cout << "Tokenizing C++ corpus..." << std::endl;
}
void DataIngestion::generate_training_sequences() {
  std::cout << "Generating training sequences..." << std::endl;
}
void DataIngestion::bpe_encode(
    const std::string &text, std::vector<int> &tokens,
    const std::unordered_map<std::string, int> &vocab) {
  std::cout << "BPE encoding..." << std::endl;
}
std::vector<std::vector<int>> DataIngestion::get_batch() {
  std::cout << "Getting batch..." << std::endl;
  return token_batches_;
}