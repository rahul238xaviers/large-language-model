#include "DataIngestion.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main() {
    std::cout << "[INFO] Starting Data Ingestion Pipeline Test" << std::endl;
    std::cout << "[INFO] Target: test_data_ingestion" << std::endl;
    std::cout << "[INFO] System: macOS arm64 (Apple Silicon)" << std::endl << std::endl;

    // Paths relative to workspace root
    std::string data_dir = "data/raw_chunks/bigcode/the-stack-v2/data/c++/";
    std::string single_file = "train-00000-of-00214.parquet";
    std::string vocab_path = "data/raw_chunks/vocabulary/cl100k_base.tiktoken";

    // Configuration
    size_t batch_size = 64;
    size_t sequence_length = 2048;
    size_t max_shard_bytes = 100 * 1024 * 1024; // 100MB

    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 1: PIPELINE CONFIGURATION" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Data Directory:       " << data_dir << std::endl;
    std::cout << "  Active Target Shard:  " << single_file << std::endl;
    std::cout << "  Tokenizer Vocabulary: " << vocab_path << std::endl;
    std::cout << "  Batch Size:           " << batch_size << std::endl;
    std::cout << "  Sequence Length:      " << sequence_length << " tokens" << std::endl;
    std::cout << "  Max Shard Cache Size: " << std::fixed << std::setprecision(2) 
              << (max_shard_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "================================================================================" << std::endl << std::endl;

    // Start timing initialization
    auto init_start = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Initializing DataIngestion and parsing BPE vocabulary..." << std::endl;
    
    DataIngestion ingestion(data_dir, single_file, max_shard_bytes, batch_size,
                            sequence_length, vocab_path);
                            
    auto init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count() / 1000.0;
    std::cout << "[INFO] Pipeline initialization completed in " << init_ms << " ms" << std::endl << std::endl;

    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 2: BATCH INGESTION LOGS" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::left << std::setw(12) << "Batch #"
              << std::setw(15) << "Batch Size"
              << std::setw(18) << "Load Latency"
              << std::setw(20) << "Token Value Range"
              << "First 10 Token IDs" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    // Test parameters
    int total_sequences = 0;
    long long total_tokens = 0;
    size_t min_sequence_size = SIZE_MAX;
    size_t max_sequence_size = 0;
    int batch_num = 0;

    auto batch_start = std::chrono::high_resolution_clock::now();

    while (true) {
        auto batch_loop_start = std::chrono::high_resolution_clock::now();
        auto batch = ingestion.get_batch();
        auto batch_loop_end = std::chrono::high_resolution_clock::now();
        
        double batch_loop_ms = std::chrono::duration_cast<std::chrono::microseconds>(batch_loop_end - batch_loop_start).count() / 1000.0;

        if (batch.empty()) {
            break;
        }

        batch_num++;
        total_sequences += batch.size();

        // Track stats
        for (const auto& seq : batch) {
            size_t seq_size = seq.size();
            total_tokens += seq_size;
            min_sequence_size = std::min(min_sequence_size, seq_size);
            max_sequence_size = std::max(max_sequence_size, seq_size);
        }

        // Print first 5 batches and then every 200,000 batches to avoid excessive spam
        bool print_detail = (batch_num <= 5) || (batch_num % 200000 == 0);

        if (print_detail) {
            // Find token range in this batch
            int min_token = INT_MAX;
            int max_token = INT_MIN;
            for (const auto& seq : batch) {
                for (int token : seq) {
                    min_token = std::min(min_token, token);
                    max_token = std::max(max_token, token);
                }
            }
            
            std::string token_range = "[" + std::to_string(min_token) + ", " + std::to_string(max_token) + "]";
            
            // Build visual token sample
            std::string token_sample = "";
            if (!batch.empty() && !batch[0].empty()) {
                int show_count = std::min(10, (int)batch[0].size());
                for (int j = 0; j < show_count; ++j) {
                    token_sample += std::to_string(batch[0][j]) + " ";
                }
                if (batch[0].size() > 10) {
                    token_sample += "...";
                }
            }

            std::cout << std::left << std::setw(12) << ("B-" + std::to_string(batch_num))
                      << std::setw(15) << (std::to_string(batch.size()) + " seq")
                      << std::setw(18) << (std::to_string(batch_loop_ms) + " ms")
                      << std::setw(20) << token_range
                      << token_sample << std::endl;
        }
    }

    auto batch_end = std::chrono::high_resolution_clock::now();
    double total_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count() / 1000.0;

    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 3: PIPELINE THROUGHPUT & PERFORMANCE SUMMARY" << std::endl;
    std::cout << "================================================================================" << std::endl;

    if (total_sequences > 0) {
        double avg_tokens_per_sequence = (double)total_tokens / total_sequences;
        double avg_sequences_per_batch = (double)total_sequences / batch_num;
        double avg_tokens_per_batch = (double)total_tokens / batch_num;
        double total_seconds = total_duration_ms / 1000.0;
        double throughput = total_tokens / total_seconds;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Total Batches Fetched:     " << batch_num << std::endl;
        std::cout << "  Total Sequences Streamed:  " << total_sequences << std::endl;
        std::cout << "  Total Tokens Processed:    " << total_tokens << std::endl;
        std::cout << "  Average Sequences / Batch: " << avg_sequences_per_batch << std::endl;
        std::cout << "  Average Tokens / Batch:    " << avg_tokens_per_batch << std::endl;
        std::cout << "  Average Tokens / Sequence: " << avg_tokens_per_sequence << std::endl;
        std::cout << "  Sequence Length Range:     [Min: " << min_sequence_size 
                  << ", Max: " << max_sequence_size << "]" << std::endl;
        std::cout << "  Total Execution Time:      " << total_duration_ms << " ms (" 
                  << total_seconds << " seconds)" << std::endl;
        std::cout << "  Pipeline Throughput:       " << std::fixed << std::setprecision(0)
                  << throughput << " tokens/second" << std::endl;
    } else {
        std::cout << "  [WARNING] No data was fetched. Check shard paths:" << std::endl;
        std::cout << "    " << data_dir << single_file << std::endl;
    }
    std::cout << "================================================================================" << std::endl;

    return (total_sequences > 0) ? 0 : 1;
}