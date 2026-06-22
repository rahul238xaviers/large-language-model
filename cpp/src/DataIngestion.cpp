/**
 * @file DataIngestion.cpp
 * @brief Implementation of the Data Ingestion Pipeline for C++ LLM Training
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * The DataIngestion class acts as a high-performance streaming data loader for
 * LLM training. Its primary goal is to feed the transformer model with batches
 * of token IDs from Parquet shards without blowing up the computer's RAM.
 *
 * Under the hood, it executes the following sequential steps:
 *
 * 1. INITIALIZATION (Constructor):
 *    - Scans the user-defined `data_dir` for all `.parquet` files.
 *    - Alphabetically sorts the file paths so shards are read in a
 * deterministic order.
 *
 * 2. STREAMING & BATCHING (get_batch):
 *    - Serves batches of shape (batch_size, sequence_length + 1) to the
 * training loop.
 *    - Tracks the current position using `current_batch_idx_`.
 *    - When the current shard runs out of tokens, it automatically loads the
 * next file.
 *
 * 3. SHARD LOADING (load_parquet_shard):
 *    - Opens the target Parquet file using Apache Arrow's memory-mapped IO.
 *    - Reads the Parquet file into an in-memory `arrow::Table` and caches it.
 *
 * 4. TOKENIZATION (tokenize_cpp_corpus):
 *    - Extracts the raw text from the specified text column (e.g., "code" or
 * "content").
 *    - Processes each text document row-by-row and translates characters into
 * token IDs via Byte Pair Encoding (BPE).
 *    - Appends the resulting token IDs to a flat memory buffer
 * (`flat_tokens_`).
 *
 * 5. SLICING (generate_training_sequences):
 *    - Slices the flat list of token IDs into individual training sequences of
 * length (sequence_length + 1). The "+1" allows us to split the sequence into
 * inputs (X) and targets (Y) shifted by 1 token.
 *
 * 6. MEMORY MANAGEMENT (shards_.clear()):
 *    - Once a Parquet shard is tokenized into integer IDs, the heavy Arrow
 * Table is immediately cleared from RAM. Only lightweight token sequences are
 * kept in memory during training, keeping the memory footprint constant and
 * safe.
 * ============================================================================
 */

#include "DataIngestion.hpp"
#include <algorithm>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/memory_pool.h>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <parquet/arrow/reader.h>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>

DataIngestion::DataIngestion(const std::string &data_dir,
                             const std::string &single_data_file,
                             size_t max_shard_bytes, size_t batch_size,
                             size_t sequence_length,
                             const std::string &vocab_path)
    : data_dir_(data_dir), single_data_file(single_data_file),
      batch_size_(batch_size), sequence_length_(sequence_length),
      max_shard_bytes_(max_shard_bytes), vocab_path_(vocab_path),
      current_batch_idx_(0), current_shard_idx_(0) {
  // Determine JSON configuration file path from .tiktoken vocab_path
  std::string json_path = vocab_path;
  size_t ext_pos = json_path.rfind(".tiktoken");
  if (ext_pos != std::string::npos) {
    json_path.replace(ext_pos, 9, ".json");
  }

  // If JSON file doesn't exist, run the python conversion script to generate it
  if (!std::filesystem::exists(json_path)) {
    if (!std::filesystem::exists(vocab_path)) {
      std::cerr << "Error: Vocab path " << vocab_path << " does not exist."
                << std::endl;
      exit(1);
    }
    std::cout << "Converting " << vocab_path
              << " to Hugging Face JSON format..." << std::endl;
    // Command to run converter: python3 cpp/scripts/convert_tiktoken.py
    // vocab_path json_path
    std::string cmd = "python3 cpp/scripts/convert_tiktoken.py " + vocab_path +
                      " " + json_path;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      std::cerr << "Error: Failed to convert tiktoken to JSON." << std::endl;
      exit(1);
    }
  }

  // Read the JSON contents into a string blob
  std::ifstream json_file(json_path, std::ios::in | std::ios::binary);
  if (!json_file.is_open()) {
    std::cerr << "Error: Failed to open tokenizer JSON at " << json_path
              << std::endl;
    exit(1);
  }
  std::string json_blob;
  json_file.seekg(0, std::ios::end);
  json_blob.resize(json_file.tellg());
  json_file.seekg(0, std::ios::beg);
  json_file.read(&json_blob[0], json_blob.size());
  json_file.close();

  // Instantiate the tokenizer
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
  if (!tokenizer_) {
    std::cerr << "Error: Failed to parse tokenizer JSON from " << json_path
              << std::endl;
    exit(1);
  }

  if (!single_data_file.empty()) {
    std::filesystem::path full_path =
        std::filesystem::path(data_dir) / single_data_file;
    if (std::filesystem::exists(full_path)) {
      file_paths_.push_back(full_path.string());
    } else {
      std::cerr << "Warning: Single data file " << full_path
                << " does not exist!" << std::endl;
    }
  } else {
    for (const auto &entry : std::filesystem::directory_iterator(data_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
        file_paths_.push_back(entry.path().string());
      }
    }
    std::sort(file_paths_.begin(), file_paths_.end());
  }
}

DataIngestion::~DataIngestion() = default;

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
        // The raw tokens are sent for tokenisation.
        bpe_encode(text, flat_tokens_);
        flat_tokens_.push_back(EOT);
      }
    }
  }
}
void DataIngestion::generate_training_sequences() {
  std::cout << "Generating training sequences..." << std::endl;

  token_batches_.clear();
  current_batch_idx_ = 0;

  size_t step = sequence_length_ + 1;

  if (flat_tokens_.size() < step) {
    return; // Not enough tokens to create sequence
  }

  for (size_t i = 0; i + step <= flat_tokens_.size(); i += step) {

    std::vector<int> sequence(flat_tokens_.begin() + i,
                              flat_tokens_.begin() + i + step);
    token_batches_.push_back(sequence);
  }
}

void DataIngestion::bpe_encode(const std::string &text,
                               std::vector<int> &tokens) {
  if (text.empty())
    return;

  std::vector<int32_t> encoded = tokenizer_->Encode(text);
  tokens.insert(tokens.end(), encoded.begin(), encoded.end());
}

std::vector<std::vector<int>> DataIngestion::get_batch() {
  std::cout << "Getting batch..." << std::endl;

  while (current_batch_idx_ + batch_size_ > token_batches_.size()) {

    if (file_paths_.empty()) {
      std::cerr << "Error: No file paths found to load" << std::endl;
      return {};
    }

    flat_tokens_.clear();

    // Remove the modulo logic, just increment and check bounds
    if (shards_.empty() && current_batch_idx_ == 0 && current_shard_idx_ == 0) {
      std::cout << "Loading shards for the first time" << std::endl;
      // Use current_shard_idx_ as is (0)
    } else {
      current_shard_idx_++;
    }

    if (current_shard_idx_ >= file_paths_.size()) {
      return {}; // No more shards
    }

    auto status = load_parquet_shard(file_paths_[current_shard_idx_]);
    if (!status.ok()) {
      std::cerr << "Error: Failed to load shard: " << status.ToString()
                << std::endl;
      current_shard_idx_++; // Skip this shard
      if (current_shard_idx_ >= file_paths_.size()) {
        return {}; // No more shards to try
      }
      continue;
    }

    tokenize_cpp_corpus(shards_.back(), "content");
    generate_training_sequences();
    shards_.clear();
    if (token_batches_.empty()) {
      continue; // This shard was too small, go to next shard
    }
  }
  std::vector<std::vector<int>> batch;
  batch.reserve(batch_size_); // Pre-allocate memory for performance

  size_t available = token_batches_.size() - current_batch_idx_;
  size_t batch_count = std::min(batch_size_, available);

  for (size_t i = 0; i < batch_count; ++i) {
    batch.push_back(token_batches_[current_batch_idx_ + i]);
  }
  current_batch_idx_ += batch_count;
  return batch;
}
