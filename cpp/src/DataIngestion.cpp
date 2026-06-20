#include "DataIngestion.hpp"
#include "algorithm"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/memory_pool.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <parquet/arrow/reader.h>

DataIngestion::DataIngestion(const std::string &data_dir,
                             size_t max_shard_bytes) {
  data_dir_ = data_dir;
  max_shard_bytes_ = max_shard_bytes;

  for (const auto &entry : std::filesystem::directory_iterator(data_dir)) {

    if (entry.is_regular_file() && entry.path().extension() == ".parquet") {

      file_paths_.push_back(entry.path().string());
    }
  }
  std::sort(file_paths_.begin(), file_paths_.end());
}

// Uses the macro functions available in the arrow C++ library
arrow::Status DataIngestion::load_parquet_shard(const std::string &file_path) {

  std::cout << "Loading shard: " << file_path << std::endl;

  // This would check if the file can be opened for reading or not
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::io::ReadableFile> infile,
                        arrow::io::ReadableFile::Open(file_path));

  // This would open the file and load the data in default memory pool
  ARROW_ASSIGN_OR_RAISE(
      std::unique_ptr<parquet::arrow::FileReader> arrow_reader,
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool()))

  // Read the file from the parquet file into a table
  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, arrow_reader->ReadTable());

  // Push the table into the shards_
  shards_.push_back(table);
  return arrow::Status::OK();
}

void DataIngestion::tokenize_cpp_corpus(std::shared_ptr<arrow::Table> table,
                                        const std::string col) {
  std::cout << "Tokenizing C++ corpus..." << std::endl;
  std::shared_ptr<arrow::ChunkedArray> chunked_array =
      table->GetColumnByName(col);

  if (!chunked_array) {
    std::cerr << "Column '" << col << "' not found!" << std::endl;
    return;
  }

  for (int i = 0; i < chunked_array->num_chunks(); i++) {
    std::shared_ptr<arrow::Array> chunk = chunked_array->chunk(i);

    auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);

    for (int64_t j = 0; j < string_array->length(); ++j) {
      if (string_array->IsValid(j)) {
        std::string text = string_array->GetString(j);
      }
    }
  }
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