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
 *      deterministic order.
 *    - Parses the OpenAI `cl100k_base.tiktoken` file directly, decodes the
 *      base64 strings, and reconstructs BPE merges in memory to generate a
 *      Hugging Face compatible BPE tokenizer configuration string in RAM.
 *    - Instantiates the `tokenizers-cpp` library via `FromBlobJSON()` to load
 * the configuration without any file system writes or subprocess calls.
 *
 * 2. STREAMING & BATCHING (get_batch):
 *    - Serves batches of shape (batch_size, sequence_length + 1) to the
 *      training loop.
 *    - Tracks the current position using `current_batch_idx_`.
 *    - When the current shard runs out of tokens, it automatically loads the
 *      next file.
 *
 * 3. SHARD LOADING (load_parquet_shard):
 *    - Opens the target Parquet file using Apache Arrow's memory-mapped IO.
 *    - Reads the Parquet file into an in-memory `arrow::Table` and caches it.
 *
 * 4. TOKENIZATION (tokenize_cpp_corpus):
 *    - Extracts the raw text from the specified text column (e.g., "content").
 *    - Processes each text document row-by-row and translates characters into
 *      token IDs using the Hugging Face Rust-based BPE tokenizer (via
 * tokenizers-cpp FFI).
 *    - Appends the resulting token IDs to a flat memory buffer
 * (`flat_tokens_`).
 *
 * 5. SLICING (generate_training_sequences):
 *    - Slices the flat list of token IDs into individual training sequences of
 *      length (sequence_length + 1). The "+1" allows us to split the sequence
 * into inputs (X) and targets (Y) shifted by 1 token.
 *
 * 6. MEMORY MANAGEMENT (shards_.clear()):
 *    - Once a Parquet shard is tokenized into integer IDs, the heavy Arrow
 *      Table is immediately cleared from RAM. Only lightweight token sequences
 * are kept in memory during training, keeping the memory footprint constant and
 *      safe.
 * ============================================================================
 */

#include "DataIngestion.hpp"
#include <cstdlib>
#include <tokenizers_cpp.h>
#include <unordered_map>
#include <thread>

namespace {

std::string base64_decode(const std::string &in) {
  std::string out;
  std::vector<int> T(256, -1);
  const std::string chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (int i = 0; i < 64; i++) {
    T[static_cast<unsigned char>(chars[i])] = i;
  }

  int val = 0, valb = -8;
  for (char c : in) {
    if (T[static_cast<unsigned char>(c)] == -1)
      break;
    val = (val << 6) + T[static_cast<unsigned char>(c)];
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<char>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

std::string bytes_to_unicode(const std::string &bytes) {
  std::string result;
  for (unsigned char b : bytes) {
    int cp = b;
    bool directly_mapped = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                           (b >= 174 && b <= 255);
    if (!directly_mapped) {
      if (b <= 32) {
        cp = 256 + b;
      } else if (b >= 127 && b <= 160) {
        cp = 256 + 33 + (b - 127);
      } else if (b == 173) {
        cp = 256 + 33 + 34; // 323
      }
    }

    if (cp < 128) {
      result.push_back(static_cast<char>(cp));
    } else {
      result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  return result;
}

std::string escape_json(const std::string &s) {
  std::string result;
  for (char c : s) {
    if (c == '\\') {
      result += "\\\\";
    } else if (c == '"') {
      result += "\\\"";
    } else if (c == '\n') {
      result += "\\n";
    } else if (c == '\r') {
      result += "\\r";
    } else if (c == '\t') {
      result += "\\t";
    } else if (static_cast<unsigned char>(c) < 32) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
      result += buf;
    } else {
      result.push_back(c);
    }
  }
  return result;
}

std::string build_tokenizer_json(const std::string &vocab_json,
                                 const std::string &merges_json) {
  std::string json = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 100257,
      "content": "<|endoftext|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 100258,
      "content": "<|fim_prefix|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 100259,
      "content": "<|fim_middle|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 100260,
      "content": "<|fim_suffix|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 100276,
      "content": "<|detokenizer_all_special_tokens|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": false,
    "use_regex": true
  },
  "post_processor": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": false,
    "use_regex": true
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": false,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "vocab": )";

  json += vocab_json;
  json += R"(,
    "merges": [
)";

  json += merges_json;
  json += R"(
    ]
  }
})";
  return json;
}

} // namespace

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
#include <vector>

/**
 * @brief Construct a new DataIngestion object.
 *
 * Scans directories for Parquet shards and loads/sorts the tiktoken vocabulary directly in C++.
 *
 * @param data_dir Path to the directory containing Parquet shards.
 * @param single_data_file Optional path to a single Parquet file to ingest.
 * @param max_shard_bytes Maximum memory size threshold of a single Parquet shard.
 * @param batch_size Number of sequences per training batch.
 * @param sequence_length Sequence token length for target predictions.
 * @param vocab_path Path to the tiktoken vocabulary file.
 */
DataIngestion::DataIngestion(const std::string &data_dir,
                             const std::string &single_data_file,
                             size_t max_shard_bytes, size_t batch_size,
                             size_t sequence_length,
                             const std::string &vocab_path)
    : data_dir_(data_dir), single_data_file(single_data_file),
      batch_size_(batch_size), sequence_length_(sequence_length),
      max_shard_bytes_(max_shard_bytes), vocab_path_(vocab_path),
      current_batch_idx_(0), current_shard_idx_(0) {

  std::cout << "Loading tiktoken vocabulary directly in C++: " << vocab_path
            << std::endl;

  std::ifstream file(vocab_path);
  if (!file.is_open()) {
    std::cerr << "Error: Failed to open vocab file at " << vocab_path
              << std::endl;
    exit(1);
  }

  std::vector<std::pair<std::string, int>> sorted_vocab;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    size_t space_idx = line.find(' ');
    if (space_idx != std::string::npos) {
      std::string base64_token = line.substr(0, space_idx);
      int rank = std::stoi(line.substr(space_idx + 1));
      std::string token_bytes = base64_decode(base64_token);
      sorted_vocab.push_back({token_bytes, rank});
    }
  }
  file.close();

  // Sort by rank
  std::sort(sorted_vocab.begin(), sorted_vocab.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  // Reconstruct merges in memory
  std::unordered_map<std::string, int> vocab_ranks_restricted;
  for (const auto &p : sorted_vocab) {
    if (p.first.length() == 1) {
      vocab_ranks_restricted[p.first] = p.second;
    }
  }

  std::vector<std::string> merges;
  for (const auto &p : sorted_vocab) {
    const std::string &token_bytes = p.first;
    int rank = p.second;
    if (token_bytes.length() <= 1)
      continue;

    // Run BPE tokenization on token_bytes using vocab_ranks_restricted
    std::vector<std::string> parts;
    for (char c : token_bytes) {
      parts.push_back(std::string(1, c));
    }
    while (parts.size() > 1) {
      int min_rank = std::numeric_limits<int>::max();
      size_t best_idx = -1;
      for (size_t i = 0; i < parts.size() - 1; i++) {
        std::string pair = parts[i] + parts[i + 1];
        auto it = vocab_ranks_restricted.find(pair);
        if (it != vocab_ranks_restricted.end() && it->second < min_rank) {
          min_rank = it->second;
          best_idx = i;
        }
      }
      if (min_rank == std::numeric_limits<int>::max()) {
        break;
      }
      parts[best_idx] = parts[best_idx] + parts[best_idx + 1];
      parts.erase(parts.begin() + best_idx + 1);
    }

    if (parts.size() == 2) {
      std::string left = bytes_to_unicode(parts[0]);
      std::string right = bytes_to_unicode(parts[1]);
      merges.push_back(left + " " + right);
    }
    vocab_ranks_restricted[token_bytes] = rank;
  }

  // Build vocab JSON
  std::string vocab_json = "{";
  for (size_t i = 0; i < sorted_vocab.size(); i++) {
    if (i > 0)
      vocab_json += ",";
    std::string hf_token = bytes_to_unicode(sorted_vocab[i].first);
    vocab_json += "\"" + escape_json(hf_token) +
                  "\":" + std::to_string(sorted_vocab[i].second);
  }
  vocab_json += "}";

  // Build merges JSON
  std::string merges_json;
  for (size_t i = 0; i < merges.size(); i++) {
    if (i > 0)
      merges_json += ",\n";
    merges_json += "      \"" + escape_json(merges[i]) + "\"";
  }

  std::string json_blob = build_tokenizer_json(vocab_json, merges_json);
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
  if (!tokenizer_) {
    std::cerr << "Error: Failed to instantiate tokenizer in memory."
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
    std::vector<std::filesystem::directory_entry> parquet_entries;
    std::copy_if(std::filesystem::directory_iterator(data_dir),
                 std::filesystem::directory_iterator(),
                 std::back_inserter(parquet_entries), [](const auto &entry) {
                   return entry.is_regular_file() &&
                          entry.path().extension() == ".parquet";
                 });

    std::transform(parquet_entries.begin(), parquet_entries.end(),
                   std::back_inserter(file_paths_),
                   [](const auto &entry) { return entry.path().string(); });

    std::sort(file_paths_.begin(), file_paths_.end());
  }
}

/**
 * @brief Destroys the DataIngestion object.
 */
DataIngestion::~DataIngestion() = default;

/**
 * @brief Loads a single Parquet shard file into Arrow memory.
 *
 * @param file_path Absolute path to the Parquet shard file.
 * @return arrow::Status Success or failure status.
 */
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

/**
 * @brief Tokenizes the specified text column of an Arrow table across all CPU cores.
 *
 * @param table Pointer to Arrow Table.
 * @param col Name of the text column to tokenize.
 */
void DataIngestion::tokenize_cpp_corpus(std::shared_ptr<arrow::Table> table,
                                        const std::string col) {
  std::cout << "Gathering documents for tokenization..." << std::endl;
  std::shared_ptr<arrow::ChunkedArray> chunked_array =
      table->GetColumnByName(col);

  if (!chunked_array) {
    std::cerr << "Column '" << col << "' not found!" << std::endl;
    return;
  }

  std::vector<std::string> documents;
  for (int i = 0; i < chunked_array->num_chunks(); i++) {
    std::shared_ptr<arrow::Array> chunk = chunked_array->chunk(i);
    auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);

    for (int64_t j = 0; j < string_array->length(); ++j) {
      if (string_array->IsValid(j)) {
        documents.push_back(string_array->GetString(j));
      }
    }
  }

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;
  std::cout << "Tokenizing C++ corpus (" << documents.size() << " docs) using " 
            << num_threads << " CPU threads..." << std::endl;

  std::vector<std::vector<int>> doc_tokens(documents.size());
  std::vector<std::thread> workers;
  
  size_t docs_per_thread = (documents.size() + num_threads - 1) / num_threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start_idx = t * docs_per_thread;
    size_t end_idx = std::min(start_idx + docs_per_thread, documents.size());

    if (start_idx >= end_idx) continue;

    workers.emplace_back([this, start_idx, end_idx, &documents, &doc_tokens]() {
      for (size_t k = start_idx; k < end_idx; ++k) {
        std::vector<int> tokens;
        bpe_encode(documents[k], tokens);
        tokens.push_back(EOT);
        doc_tokens[k] = std::move(tokens);
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  std::cout << "Flattening tokens..." << std::endl;
  for (auto &tokens : doc_tokens) {
    flat_tokens_.insert(flat_tokens_.end(), tokens.begin(), tokens.end());
  }
}
/**
 * @brief Cuts the flat token stream into sequential chunks of sequence_length + 1.
 */
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

/**
 * @brief Encodes raw text into token IDs using BPE.
 *
 * @param text Input raw string.
 * @param tokens Vector to insert encoded token IDs into.
 */
void DataIngestion::bpe_encode(const std::string &text,
                               std::vector<int> &tokens) {
  if (text.empty())
    return;

  std::vector<int32_t> encoded = tokenizer_->Encode(text);
  tokens.insert(tokens.end(), encoded.begin(), encoded.end());
}

/**
 * @brief Retrieves the next batch of training sequences, loading new shards as needed.
 *
 * @return std::vector<std::vector<int>> A batch of token lists.
 */
std::vector<std::vector<int>> DataIngestion::get_batch() {
  // std::cout << "Getting batch..." << std::endl;

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

  batch.assign(token_batches_.begin() + current_batch_idx_,
               token_batches_.begin() + current_batch_idx_ + batch_count);
  current_batch_idx_ += batch_count;
  return batch;
}

void DataIngestion::skip_sequences(size_t num_sequences) {
  size_t skipped = 0;
  while (skipped < num_sequences) {
    if (current_batch_idx_ >= token_batches_.size()) {
      if (file_paths_.empty()) {
        break;
      }
      flat_tokens_.clear();
      if (shards_.empty() && current_batch_idx_ == 0 && current_shard_idx_ == 0) {
        // First time loading shards
      } else {
        current_shard_idx_++;
      }
      if (current_shard_idx_ >= file_paths_.size()) {
        break;
      }
      auto status = load_parquet_shard(file_paths_[current_shard_idx_]);
      if (!status.ok()) {
        current_shard_idx_++;
        continue;
      }
      tokenize_cpp_corpus(shards_.back(), "content");
      generate_training_sequences();
      shards_.clear();
      if (token_batches_.empty()) {
        continue;
      }
    }
    size_t available = token_batches_.size() - current_batch_idx_;
    size_t to_skip = std::min(num_sequences - skipped, available);
    current_batch_idx_ += to_skip;
    skipped += to_skip;
  }
  std::cout << "[INFO] Resume | Skipped " << skipped << " already-processed sequences (" 
            << skipped * (sequence_length_ + 1) << " tokens) from data stream." << std::endl;
}
