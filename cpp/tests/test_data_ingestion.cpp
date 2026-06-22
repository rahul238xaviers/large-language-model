#include "DataIngestion.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "   Data Ingestion Performance Test" << std::endl;
  std::cout << "========================================" << std::endl
            << std::endl;

  // Paths relative to workspace root
  std::string data_dir = "data/raw_chunks/bigcode/the-stack-v2/data/c++/";
  std::string single_file = "train-00000-of-00214.parquet";
  std::string vocab_path = "data/raw_chunks/vocabulary/cl100k_base.tiktoken";

  // Configuration
  size_t batch_size = 2;
  size_t sequence_length = 10;
  size_t max_shard_bytes = 10 * 1024 * 1024; // 10MB

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Data directory: " << data_dir << std::endl;
  std::cout << "  Single file: " << single_file << std::endl;
  std::cout << "  Vocabulary: " << vocab_path << std::endl;
  std::cout << "  Batch size: " << batch_size << std::endl;
  std::cout << "  Sequence length: " << sequence_length << std::endl;
  std::cout << "  Max shard bytes: " << max_shard_bytes << " bytes"
            << std::endl;
  std::cout << std::endl;

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();

  // Instantiate DataIngestion
  std::cout << "Initializing DataIngestion..." << std::endl;
  DataIngestion ingestion(data_dir, single_file, max_shard_bytes, batch_size,
                          sequence_length, vocab_path);
  std::cout << "✓ Initialization complete" << std::endl << std::endl;

  // Test parameters
  const int num_batches_to_fetch = 5;
  int total_sequences = 0;
  int total_tokens = 0;
  size_t min_sequence_size = SIZE_MAX;
  size_t max_sequence_size = 0;

  std::cout << "Fetching " << num_batches_to_fetch << " batches..."
            << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  auto batch_start = std::chrono::high_resolution_clock::now();

  for (int batch_num = 0; batch_num < num_batches_to_fetch; ++batch_num) {
    auto batch_loop_start = std::chrono::high_resolution_clock::now();

    std::cout << "\nBatch " << batch_num + 1 << ":" << std::endl;

    auto batch = ingestion.get_batch();

    auto batch_loop_end = std::chrono::high_resolution_clock::now();
    auto batch_loop_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(batch_loop_end -
                                                              batch_loop_start);

    if (batch.empty()) {
      std::cout << "  ⚠ No more batches available (end of data)" << std::endl;
      break;
    }

    std::cout << "  Size: " << batch.size() << " sequences" << std::endl;
    std::cout << "  Load time: " << batch_loop_duration.count() << " ms"
              << std::endl;

    total_sequences += batch.size();

    // Analyze each sequence in the batch
    for (size_t i = 0; i < batch.size(); ++i) {
      size_t seq_size = batch[i].size();
      total_tokens += seq_size;
      min_sequence_size = std::min(min_sequence_size, seq_size);
      max_sequence_size = std::max(max_sequence_size, seq_size);

      // Print first 20 tokens of first sequence only (to avoid spam)
      if (i == 0) {
        std::cout << "  First sequence (" << seq_size << " tokens): ";
        int tokens_to_show = std::min(20, (int)seq_size);
        for (int j = 0; j < tokens_to_show; ++j) {
          std::cout << batch[i][j] << " ";
        }
        if (seq_size > 20)
          std::cout << "...";
        std::cout << std::endl;
      }
    }

    // Print token range in this batch
    int min_token = INT_MAX;
    int max_token = INT_MIN;
    for (const auto &seq : batch) {
      for (int token : seq) {
        min_token = std::min(min_token, token);
        max_token = std::max(max_token, token);
      }
    }
    std::cout << "  Token range: [" << min_token << ", " << max_token << "]"
              << std::endl;
  }

  auto batch_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      batch_end - start_time);

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "           PERFORMANCE SUMMARY" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  if (total_sequences > 0) {
    double avg_tokens_per_sequence = (double)total_tokens / total_sequences;
    double avg_sequences_per_batch =
        (double)total_sequences / num_batches_to_fetch;
    double avg_tokens_per_batch = (double)total_tokens / num_batches_to_fetch;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total batches fetched: " << num_batches_to_fetch
              << std::endl;
    std::cout << "  Total sequences: " << total_sequences << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Average sequences per batch: " << avg_sequences_per_batch
              << std::endl;
    std::cout << "  Average tokens per batch: " << avg_tokens_per_batch
              << std::endl;
    std::cout << "  Average tokens per sequence: " << avg_tokens_per_sequence
              << std::endl;
    std::cout << "  Min sequence size: " << min_sequence_size << std::endl;
    std::cout << "  Max sequence size: " << max_sequence_size << std::endl;
    std::cout << "  Total time: " << total_duration.count() << " ms"
              << std::endl;
    std::cout << "  Throughput: "
              << (total_tokens * 1000.0 / total_duration.count())
              << " tokens/second" << std::endl;
  } else {
    std::cout << "  ⚠ No data was fetched!" << std::endl;
    std::cout << "  Check that your data path is correct:" << std::endl;
    std::cout << "    " << data_dir << single_file << std::endl;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "Test completed!" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}