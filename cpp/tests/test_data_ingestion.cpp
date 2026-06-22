#include "DataIngestion.hpp"
#include <iostream>

int main() {
  std::cout << "Starting Data Ingestion test..." << std::endl;

  // Paths relative to workspace root
  std::string data_dir = "data/raw_chunks/bigcode/the-stack-v2/data/c++/";
  std::string single_file = "train-00000-of-00214.parquet";
  std::string vocab_path = "data/raw_chunks/vocabulary/cl100k_base.tiktoken";

  // Instantiation arguments:
  // data_dir, max_shard_bytes (10MB), batch_size (2), sequence_length (10),
  // vocab_path
  DataIngestion ingestion(data_dir, single_file, 10 * 1024 * 1024, 2, 10,
                          vocab_path);

  std::cout << "Getting first batch..." << std::endl;
  auto batch = ingestion.get_batch();

  std::cout << "Retrieved batch of size: " << batch.size() << std::endl;
  for (size_t i = 0; i < batch.size(); ++i) {
    std::cout << "Sequence " << i << ": ";
    for (int token : batch[i]) {
      std::cout << token << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Test completed successfully!" << std::endl;
  return 0;
}
