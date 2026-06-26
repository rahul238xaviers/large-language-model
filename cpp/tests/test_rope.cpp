#include "Tensor.hpp"
#include "Positional.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <numeric>

// Formatting helper for test rows
void print_test_row(const std::string& id, const std::string& desc, 
                    const std::string& expected, const std::string& actual, 
                    bool passed) {
    std::cout << std::left << std::setw(10) << id
              << std::setw(38) << desc
              << std::setw(20) << expected
              << std::setw(20) << actual
              << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
              << std::endl;
}

// Helper to check if two floats are approximately equal
bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[INFO] Starting RoPE Layer Test Suite" << std::endl;
    std::cout << "[INFO] Target: test_rope" << std::endl;
    std::cout << "[INFO] System: macOS arm64 (Apple Silicon)" << std::endl << std::endl;

    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 1: FUNCTIONAL & UNIT VERIFICATION" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::left << std::setw(10) << "Test ID"
              << std::setw(38) << "Test Description"
              << std::setw(20) << "Expected Value"
              << std::setw(20) << "Actual Value"
              << "Status" << std::endl;
    std::cout << std::string(96, '-') << std::endl;

    int passed_checks = 0;
    int total_checks = 0;

    // Parameters for testing: head_dim = 8, max_seq_len = 10, base = 10000.0f
    size_t test_head_dim = 8;
    size_t test_max_seq = 10;
    float test_base = 10000.0f;
    RoPE rope(test_head_dim, test_max_seq, test_base);

    // TC-01: Verify precomputed table shapes
    {
        total_checks++;
        const auto& cos_t = rope.cos_table();
        const auto& sin_t = rope.sin_table();
        bool pass = (cos_t.size() == test_max_seq) && (cos_t[0].size() == test_head_dim / 2) &&
                    (sin_t.size() == test_max_seq) && (sin_t[0].size() == test_head_dim / 2);
        if (pass) passed_checks++;
        print_test_row("TC-01", "Precomputed table dimensions", 
                       "[" + std::to_string(test_max_seq) + ", " + std::to_string(test_head_dim / 2) + "]",
                       "[" + std::to_string(cos_t.size()) + ", " + std::to_string(cos_t.empty() ? 0 : cos_t[0].size()) + "]", 
                       pass);
    }

    // TC-02: Verify pos = 0 values (cos should be 1, sin should be 0)
    {
        total_checks++;
        bool pass = true;
        const auto& cos_t = rope.cos_table();
        const auto& sin_t = rope.sin_table();
        for (size_t i = 0; i < test_head_dim / 2; ++i) {
            if (!approx_equal(cos_t[0][i], 1.0f) || !approx_equal(sin_t[0][i], 0.0f)) {
                pass = false;
            }
        }
        if (pass) passed_checks++;
        print_test_row("TC-02", "Lookup table at pos=0 (identity)", "cos=1.0, sin=0.0", 
                       "cos=" + std::to_string(cos_t[0][0]) + ", sin=" + std::to_string(sin_t[0][0]), pass);
    }

    // TC-03: Verify pos = 1 value correctness at i = 0
    {
        total_checks++;
        // For i=0, theta_0 = 1 / 10000^0 = 1.0f. Angle at pos=1 is 1.0f.
        float expected_cos = std::cos(1.0f);
        float expected_sin = std::sin(1.0f);
        const auto& cos_t = rope.cos_table();
        const auto& sin_t = rope.sin_table();
        bool pass = approx_equal(cos_t[1][0], expected_cos) && approx_equal(sin_t[1][0], expected_sin);
        if (pass) passed_checks++;
        print_test_row("TC-03", "Precomputation accuracy at pos=1, i=0",
                       "cos=" + std::to_string(expected_cos) + ", sin=" + std::to_string(expected_sin),
                       "cos=" + std::to_string(cos_t[1][0]) + ", sin=" + std::to_string(sin_t[1][0]), pass);
    }

    // TC-04: Rotation at pos = 0 (should be identity)
    {
        total_checks++;
        Tensor q({1, 1, 1, 8}, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}));
        Tensor k({1, 1, 1, 8}, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}));
        rope.forward(q, k);
        bool pass = true;
        for (size_t d = 0; d < 8; ++d) {
            if (!approx_equal(q(0, 0, 0, d), static_cast<float>(d + 1))) pass = false;
        }
        if (pass) passed_checks++;
        print_test_row("TC-04", "In-place rotation at pos=0 (identity)", "1.00000 at idx 0", 
                       std::to_string(q(0, 0, 0, 0)), pass);
    }

    // TC-05: Rotation at pos = 1 correctness (pairs: (1, 2) rotated by angle 1.0f)
    {
        total_checks++;
        Tensor q({1, 1, 2, 8}, std::vector<float>({
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, // pos = 0
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f  // pos = 1
        }));
        Tensor k({1, 1, 2, 8}, std::vector<float>({
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, // pos = 0
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f  // pos = 1
        }));
        rope.forward(q, k);
        
        // For pos=1, i=0: x0 = 1.0f, x1 = 2.0f. Angle = 1.0f.
        float c = std::cos(1.0f);
        float s = std::sin(1.0f);
        float expected_q_1_0 = 1.0f * c - 2.0f * s; // x0*cos - x1*sin
        float actual_q_1_0 = q(0, 0, 1, 0);
        bool pass = approx_equal(actual_q_1_0, expected_q_1_0);
        if (pass) passed_checks++;
        print_test_row("TC-05", "Rotation correctness at pos=1, pair 0",
                       std::to_string(expected_q_1_0), std::to_string(actual_q_1_0), pass);
    }

    // TC-06: Error handling - non-4D inputs
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor wrong_q({1, 1, 8});
            Tensor wrong_k({1, 1, 8});
            rope.forward(wrong_q, wrong_k);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-06", "Error Case: Non-4D Tensors input", "Throw std::invalid_arg", 
                       pass ? "Exception Thrown" : "No Exception", pass);
    }

    // TC-07: Error handling - head dimension mismatch
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor wrong_q({1, 1, 1, 6});
            Tensor wrong_k({1, 1, 1, 6});
            rope.forward(wrong_q, wrong_k);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-07", "Error Case: Mismatched head dimension", "Throw std::invalid_arg", 
                       pass ? "Exception Thrown" : "No Exception", pass);
    }

    // TC-08: Error handling - seq_len exceeding max_seq_len
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor wrong_q({1, 1, 11, 8});
            Tensor wrong_k({1, 1, 11, 8});
            rope.forward(wrong_q, wrong_k);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-08", "Error Case: SeqLen > MaxSeqLen", "Throw std::invalid_arg", 
                       pass ? "Exception Thrown" : "No Exception", pass);
    }

    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
    std::cout << "================================================================================" << std::endl;

    // Benchmark GQA shape: Q: [64, 16, 2048, 128], K: [64, 8, 2048, 128]
    {
        std::cout << "RoPE GQA Batch Benchmark:" << std::endl;
        std::cout << "  Query Dimensions:     [64, 16, 2048, 128]" << std::endl;
        std::cout << "  Key Dimensions:       [64, 8, 2048, 128]" << std::endl;

        RoPE bench_rope(128, 2048, 10000.0f);
        
        // Size: Q has 64 * 16 * 2048 * 128 = 268,435,456 floats (1073.74 MB)
        //       K has 64 * 8 * 2048 * 128 = 134,217,728 floats (536.87 MB)
        // Total memory volume read and written per iteration = 2 * (1073.74 MB + 536.87 MB) = 3.221 GB
        double total_bytes = 2.0 * (64.0 * 16.0 * 2048.0 * 128.0 + 64.0 * 8.0 * 2048.0 * 128.0) * sizeof(float);
        double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);

        std::cout << "  Total Memory Volume:  " << std::fixed << std::setprecision(2) 
                  << (total_bytes / (1024.0 * 1024.0)) << " MB (" << total_gb << " GB)" << std::endl;

        // Initialize tensors with dummy data
        std::cout << "  [INFO] Allocating memory..." << std::endl;
        Tensor q({64, 16, 2048, 128}, 0.5f);
        Tensor k({64, 8, 2048, 128}, 0.5f);
        std::cout << "  [INFO] Memory allocated. Running benchmark..." << std::endl;

        // Warm up pass
        bench_rope.forward(q, k);

        const int runs = 5;
        std::vector<double> latencies;
        latencies.reserve(runs);

        for (int r = 0; r < runs; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            bench_rope.forward(q, k);
            auto end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            latencies.push_back(duration_ms);
            std::cout << "    Run " << (r + 1) << ": " << duration_ms << " ms" << std::endl;
        }

        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double avg_ms = sum / runs;
        double avg_seconds = avg_ms / 1000.0;
        double bandwidth = total_gb / avg_seconds;

        std::cout << "  Average Latency:      " << avg_ms << " ms" << std::endl;
        std::cout << "  Memory Bandwidth:     " << bandwidth << " GB/s" << std::endl << std::endl;
    }

    std::cout << "================================================================================" << std::endl;
    std::cout << "TEST EXECUTION SUMMARY" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Total Functional Checks: " << total_checks << std::endl;
    std::cout << "  Passed Checks:           " << passed_checks << std::endl;
    std::cout << "  Failed Checks:           " << (total_checks - passed_checks) << std::endl;
    std::cout << "  Status:                  " << (passed_checks == total_checks ? "\033[32mSUCCESS\033[0m" : "\033[31mFAILURE\033[0m") << std::endl;
    std::cout << "================================================================================" << std::endl;

    return (passed_checks == total_checks) ? 0 : 1;
}
