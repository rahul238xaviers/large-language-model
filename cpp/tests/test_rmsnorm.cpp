#include "Tensor.hpp"
#include "RMSNorm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <iomanip>
#include <string>

// Formatting helper for test rows
void print_test_row(const std::string& id, const std::string& desc, 
                    const std::string& expected, const std::string& actual, 
                    bool passed) {
    std::cout << std::left << std::setw(10) << id
              << std::setw(36) << desc
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
    std::cout << "[INFO] Starting RMSNorm Layer Test Suite" << std::endl;
    std::cout << "[INFO] Target: test_rmsnorm" << std::endl;
    std::cout << "[INFO] System: macOS arm64 (Apple Silicon)" << std::endl << std::endl;

    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 1: FUNCTIONAL & UNIT VERIFICATION" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::left << std::setw(10) << "Test ID"
              << std::setw(36) << "Test Description"
              << std::setw(20) << "Expected Value"
              << std::setw(20) << "Actual Value"
              << "Status" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    int passed_checks = 0;
    int total_checks = 0;

    // Set up test inputs
    Tensor x({2, 2}, std::vector<float>({0.3f, 0.9f, -0.7f, 0.4f}));
    RMSNorm norm(2);

    // TC-01: Weights shape
    {
        total_checks++;
        bool pass = (norm.weight().shape() == std::vector<size_t>({2}));
        if (pass) passed_checks++;
        print_test_row("TC-01", "Initial weights shape check", "[2]", 
                       "[" + std::to_string(norm.weight().shape()[0]) + "]", pass);
    }

    // TC-02: Identity initial values
    {
        total_checks++;
        bool pass = approx_equal(norm.weight().data()[0], 1.0f) && 
                    approx_equal(norm.weight().data()[1], 1.0f);
        if (pass) passed_checks++;
        print_test_row("TC-02", "Initial weights identity verification", "1.00000", 
                       std::to_string(norm.weight().data()[0]), pass);
    }

    // Run forward pass
    Tensor y = norm.forward(x);

    // TC-03 to TC-06: Value correctness
    float expected_y00 = 0.447208f;
    float expected_y01 = 1.341625f;
    float expected_y10 = -1.22786f;
    float expected_y11 = 0.701635f;

    {
        total_checks++;
        bool pass = approx_equal(y(0, 0), expected_y00);
        if (pass) passed_checks++;
        print_test_row("TC-03", "Norm correctness (Token 0, Dim 0)", 
                       std::to_string(expected_y00), std::to_string(y(0, 0)), pass);
    }
    {
        total_checks++;
        bool pass = approx_equal(y(0, 1), expected_y01);
        if (pass) passed_checks++;
        print_test_row("TC-04", "Norm correctness (Token 0, Dim 1)", 
                       std::to_string(expected_y01), std::to_string(y(0, 1)), pass);
    }
    {
        total_checks++;
        bool pass = approx_equal(y(1, 0), expected_y10);
        if (pass) passed_checks++;
        print_test_row("TC-05", "Norm correctness (Token 1, Dim 0)", 
                       std::to_string(expected_y10), std::to_string(y(1, 0)), pass);
    }
    {
        total_checks++;
        bool pass = approx_equal(y(1, 1), expected_y11);
        if (pass) passed_checks++;
        print_test_row("TC-06", "Norm correctness (Token 1, Dim 1)", 
                       std::to_string(expected_y11), std::to_string(y(1, 1)), pass);
    }

    // Custom scaling weight test
    norm.weight().data()[0] = 2.0f;
    norm.weight().data()[1] = 0.5f;
    Tensor y_scaled = norm.forward(x);
    float expected_scaled00 = expected_y00 * 2.0f;
    float expected_scaled01 = expected_y01 * 0.5f;

    {
        total_checks++;
        bool pass = approx_equal(y_scaled(0, 0), expected_scaled00);
        if (pass) passed_checks++;
        print_test_row("TC-07", "Custom scaling weight (Dim 0)", 
                       std::to_string(expected_scaled00), std::to_string(y_scaled(0, 0)), pass);
    }
    {
        total_checks++;
        bool pass = approx_equal(y_scaled(0, 1), expected_scaled01);
        if (pass) passed_checks++;
        print_test_row("TC-08", "Custom scaling weight (Dim 1)", 
                       std::to_string(expected_scaled01), std::to_string(y_scaled(0, 1)), pass);
    }

    // Boundary & error checks
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor flat({4}, 1.0f);
            norm.forward(flat);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-09", "Error handling: 1D Tensor", "Throw exception", 
                       pass ? "Exception Thrown" : "No Exception", pass);
    }
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor wrong_dim({2, 3}, 1.0f);
            norm.forward(wrong_dim);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-10", "Error handling: Dimension mismatch", "Throw exception", 
                       pass ? "Exception Thrown" : "No Exception", pass);
    }

    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARK" << std::endl;
    std::cout << "================================================================================" << std::endl;
    
    // Config large shapes representing typical 1.6B model layer activations
    size_t batch = 4;
    size_t seq_len = 2048;
    size_t hidden_dim = 2048;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Tensor Dimensions:    [" << batch << ", " << seq_len << ", " << hidden_dim << "]" << std::endl;
    
    double total_floats = static_cast<double>(batch * seq_len * hidden_dim);
    double footprint_mb = (total_floats * sizeof(float)) / (1024.0 * 1024.0);
    double traffic_mb = 2.0 * footprint_mb; // Read input + Write output
    
    std::cout << "  Total Float Elements: " << static_cast<size_t>(total_floats) << std::endl;
    std::cout << "  Memory Footprint:     " << std::fixed << std::setprecision(2) << footprint_mb << " MB" << std::endl;
    std::cout << "  Memory Traffic:       " << traffic_mb << " MB (Read + Write)" << std::endl;

    // Warmup loops
    Tensor large_x({batch, seq_len, hidden_dim}, 0.5f);
    RMSNorm large_norm(hidden_dim);
    for (int i = 0; i < 5; ++i) {
        Tensor tmp = large_norm.forward(large_x);
    }

    // Profiling runs
    const int runs = 50;
    std::vector<double> latencies;
    latencies.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor out = large_norm.forward(large_x);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        latencies.push_back(duration_ms);
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg_latency = sum / runs;
    
    double min_latency = latencies[0];
    double max_latency = latencies[0];
    for (double lat : latencies) {
        if (lat < min_latency) min_latency = lat;
        if (lat > max_latency) max_latency = lat;
    }

    double accum = 0.0;
    for (double lat : latencies) {
        accum += (lat - avg_latency) * (lat - avg_latency);
    }
    double std_dev = std::sqrt(accum / (runs - 1));

    // Memory Bandwidth in GB/s
    double avg_seconds = avg_latency / 1000.0;
    double throughput_gb = (traffic_mb / 1024.0) / avg_seconds;

    std::cout << std::endl << "Profiling Results (averaged over " << runs << " runs):" << std::endl;
    std::cout << "  Average Latency:      " << avg_latency << " ms" << std::endl;
    std::cout << "  Minimum Latency:      " << min_latency << " ms" << std::endl;
    std::cout << "  Maximum Latency:      " << max_latency << " ms" << std::endl;
    std::cout << "  Std Dev Latency:      " << std_dev << " ms" << std::endl;
    std::cout << "  Memory Bandwidth:     " << throughput_gb << " GB/s" << std::endl;

    std::cout << std::endl;
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
