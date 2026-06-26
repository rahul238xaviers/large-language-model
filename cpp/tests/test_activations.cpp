#include "Tensor.hpp"
#include "Activations.hpp"
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
    std::cout << "[INFO] Starting Activations Layer Test Suite" << std::endl;
    std::cout << "[INFO] Target: test_activations" << std::endl;
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

    // TC-01: silu_ correctness
    {
        total_checks++;
        Tensor x({2}, std::vector<float>({0.0f, 2.0f}));
        // Use the namespace typo 'activatations' to match the user's header configuration
        activatations::silu_(x);
        
        // Expected: Swish(0) = 0.0, Swish(2) = 2.0 * Sigmoid(2) = 1.76159
        bool pass = approx_equal(x.data()[0], 0.0f) && approx_equal(x.data()[1], 1.76159f);
        if (pass) passed_checks++;
        print_test_row("TC-01", "In-place SiLU (Swish) correctness", "[0.00000, 1.76159]",
                       "[" + std::to_string(x.data()[0]) + ", " + std::to_string(x.data()[1]) + "]", pass);
    }

    // TC-02: swiglu correctness
    {
        total_checks++;
        Tensor gate({2}, std::vector<float>({0.0f, 2.0f}));
        Tensor up({2}, std::vector<float>({10.0f, 5.0f}));
        
        Tensor y = activatations::swiglu(gate, up);
        
        // Expected: Swish(gate) * up = [0.0 * 10.0, 1.76159 * 5.0] = [0.0, 8.80797]
        bool pass = approx_equal(y.data()[0], 0.0f) && approx_equal(y.data()[1], 8.80797f);
        if (pass) passed_checks++;
        print_test_row("TC-02", "SwiGLU gating correctness", "[0.00000, 8.80797]",
                       "[" + std::to_string(y.data()[0]) + ", " + std::to_string(y.data()[1]) + "]", pass);
    }

    // TC-03: Shape mismatch
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor gate({2}, 1.0f);
            Tensor up({3}, 1.0f);
            activatations::swiglu(gate, up);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-03", "SwiGLU input dimension mismatch", "Throw exception",
                       pass ? "Exception Thrown" : "No Exception", pass);
    }

    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARK" << std::endl;
    std::cout << "================================================================================" << std::endl;

    // Config large intermediate layer shapes representing FFN hidden dimension
    size_t batch = 64;
    size_t seq_len = 2048;
    size_t hidden_dim = 5461; // 8/3 * 2048

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Tensor Dimensions:    [" << batch << ", " << seq_len << ", " << hidden_dim << "]" << std::endl;

    double total_floats = static_cast<double>(batch * seq_len * hidden_dim);
    double footprint_mb = (total_floats * sizeof(float)) / (1024.0 * 1024.0);
    double traffic_mb = 3.0 * footprint_mb; // Read gate + Read up + Write output

    std::cout << "  Total Float Elements: " << static_cast<size_t>(total_floats) << std::endl;
    std::cout << "  Memory Footprint:     " << std::fixed << std::setprecision(2) << footprint_mb << " MB" << std::endl;
    std::cout << "  Memory Traffic:       " << traffic_mb << " MB (Read + Read + Write)" << std::endl;

    // Initialize large inputs
    Tensor large_gate({batch, seq_len, hidden_dim}, 0.5f);
    Tensor large_up({batch, seq_len, hidden_dim}, 1.5f);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        Tensor tmp = activatations::swiglu(large_gate, large_up);
    }

    // Profiling runs
    const int runs = 5;
    std::vector<double> latencies;
    latencies.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor out = activatations::swiglu(large_gate, large_up);
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

    double avg_seconds = avg_latency / 1000.0;
    double throughput_gb = (traffic_mb / 1024.0) / avg_seconds;

    std::cout << std::endl << "Profiling Results (averaged over " << runs << " runs):" << std::endl;
    std::cout << "  Average Latency:      " << avg_latency << " ms" << std::endl;
    std::cout << "  Minimum Latency:      " << min_latency << " ms" << std::endl;
    std::cout << "  Maximum Latency:      " << max_latency << " ms" << std::endl;
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
