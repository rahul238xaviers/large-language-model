#include "Tensor.hpp"
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
bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "[INFO] Starting Tensor Operations Test Suite" << std::endl;
    std::cout << "[INFO] Target: test_math_ops" << std::endl;
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

    // 1. Default construction
    {
        total_checks++;
        Tensor t0;
        bool pass = t0.shape().empty() && (t0.size() == 0 || t0.size() == 1);
        if (pass) passed_checks++;
        print_test_row("TC-01", "Default construction shape check", "0 (Empty)", 
                       std::to_string(t0.shape().size()), pass);
    }

    // 2. Shape construction
    Tensor t1({2, 3});
    {
        total_checks++;
        bool pass = (t1.shape() == std::vector<size_t>({2, 3})) && (t1.size() == 6);
        if (pass) passed_checks++;
        print_test_row("TC-02", "Shape constructor dimensions", "[2, 3] size=6", 
                       "[" + std::to_string(t1.shape()[0]) + ", " + std::to_string(t1.shape()[1]) + "] size=" + std::to_string(t1.size()), pass);
    }
    {
        total_checks++;
        bool pass = (t1.strides() == std::vector<size_t>({3, 1}));
        if (pass) passed_checks++;
        print_test_row("TC-03", "Shape constructor strides", "[3, 1]", 
                       "[" + std::to_string(t1.strides()[0]) + ", " + std::to_string(t1.strides()[1]) + "]", pass);
    }

    // 3. Fill construction
    Tensor t2({2, 2}, 5.5f);
    {
        total_checks++;
        bool pass = true;
        for (size_t i = 0; i < t2.size(); ++i) {
            if (t2.data()[i] != 5.5f) pass = false;
        }
        if (pass) passed_checks++;
        print_test_row("TC-04", "Fill constructor defaults", "5.50000", 
                       std::to_string(t2.data()[0]), pass);
    }

    // 4. Data construction
    std::vector<float> custom_data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t3({2, 2}, custom_data);
    {
        total_checks++;
        bool pass = (t3.num_elements() == custom_data.size()) && 
                     std::equal(t3.data(), t3.data() + t3.num_elements(), custom_data.begin());
        if (pass) passed_checks++;
        print_test_row("TC-05", "Data constructor correctness", "4.00000 at [1,1]", 
                       std::to_string(t3.data()[3]), pass);
    }

    // 5. Indexing Fast 3D
    Tensor t4({2, 3, 4});
    float val = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                t4(i, j, k) = val;
                val += 1.0f;
            }
        }
    }
    {
        total_checks++;
        bool pass = approx_equal(t4(1, 1, 1), 17.0f);
        if (pass) passed_checks++;
        print_test_row("TC-06", "Fast 3D index write/read", "17.00000", 
                       std::to_string(t4(1, 1, 1)), pass);
    }
    {
        total_checks++;
        bool pass = approx_equal(t4({1, 1, 1}), 17.0f);
        if (pass) passed_checks++;
        print_test_row("TC-07", "Multi-dim vector indexing", "17.00000", 
                       std::to_string(t4({1, 1, 1})), pass);
    }

    // 6. Element-wise math
    Tensor A({2, 2}, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    Tensor B({2, 2}, std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}));
    {
        total_checks++;
        Tensor C = A.add(B);
        std::vector<float> expC({6.0f, 8.0f, 10.0f, 12.0f});
        bool pass = std::equal(C.data(), C.data() + C.num_elements(), expC.begin());
        if (pass) passed_checks++;
        print_test_row("TC-08", "Out-of-place addition correctness", "12.00000 at idx 3", 
                       std::to_string(C.data()[3]), pass);
    }
    {
        total_checks++;
        Tensor A_copy = A;
        A_copy.add_(B);
        std::vector<float> expAC({6.0f, 8.0f, 10.0f, 12.0f});
        bool pass = std::equal(A_copy.data(), A_copy.data() + A_copy.num_elements(), expAC.begin());
        if (pass) passed_checks++;
        print_test_row("TC-09", "In-place addition correctness", "12.00000 at idx 3", 
                       std::to_string(A_copy.data()[3]), pass);
    }
    {
        total_checks++;
        Tensor D = A.mul(B);
        std::vector<float> expD({5.0f, 12.0f, 21.0f, 32.0f});
        bool pass = std::equal(D.data(), D.data() + D.num_elements(), expD.begin());
        if (pass) passed_checks++;
        print_test_row("TC-10", "Out-of-place multiply correctness", "32.00000 at idx 3", 
                       std::to_string(D.data()[3]), pass);
    }
    {
        total_checks++;
        Tensor E = A.scale(2.5f);
        std::vector<float> expE({2.5f, 5.0f, 7.5f, 10.0f});
        bool pass = std::equal(E.data(), E.data() + E.num_elements(), expE.begin());
        if (pass) passed_checks++;
        print_test_row("TC-11", "Scalar scaling correctness", "10.00000 at idx 3", 
                       std::to_string(E.data()[3]), pass);
    }

    // 7. Matmul 2D and 3D Batched
    Tensor M1({2, 3}, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    Tensor M2({3, 2}, std::vector<float>({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
    {
        total_checks++;
        Tensor M3 = M1.matmul(M2);
        bool pass = M3.shape() == std::vector<size_t>({2, 2}) &&
                    approx_equal(M3(0, 0), 58.0f) && approx_equal(M3(0, 1), 64.0f) &&
                    approx_equal(M3(1, 0), 139.0f) && approx_equal(M3(1, 1), 154.0f);
        if (pass) passed_checks++;
        print_test_row("TC-12", "2D matrix multiplication correctness", "154.00000 at [1,1]", 
                       std::to_string(M3(1, 1)), pass);
    }
    Tensor MB1({2, 2, 3}, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}));
    Tensor MB2({2, 3, 2}, std::vector<float>({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    {
        total_checks++;
        Tensor MB3 = MB1.matmul(MB2);
        bool pass = MB3.shape() == std::vector<size_t>({2, 2, 2}) &&
                    approx_equal(MB3(0, 0, 0), 58.0f) && approx_equal(MB3(1, 1, 1), 24.0f);
        if (pass) passed_checks++;
        print_test_row("TC-13", "Batched 3D matmul correctness", "24.00000 at [1,1,1]", 
                       std::to_string(MB3(1, 1, 1)), pass);
    }

    // 8. Edge cases exception handling
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor wrong_bounds({2, 2});
            wrong_bounds({2, 0});
        } catch (const std::out_of_range&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-14", "Edge Case: Out of bounds", "Throw std::out_of_range", 
                       pass ? "Thrown" : "No Exception", pass);
    }
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor wrong_dims({2, 2});
            wrong_dims({0, 0, 0});
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-15", "Edge Case: Dimensionality mismatch", "Throw std::inv_arg", 
                       pass ? "Thrown" : "No Exception", pass);
    }
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor wrong_add({2, 2});
            Tensor wrong_add_diff({2, 3});
            wrong_add.add(wrong_add_diff);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-16", "Edge Case: Add shape mismatch", "Throw std::inv_arg", 
                       pass ? "Thrown" : "No Exception", pass);
    }
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor matmul_a({2, 2, 3});
            Tensor matmul_b({3, 3, 2});
            matmul_a.matmul(matmul_b);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-17", "Edge Case: Matmul batch mismatch", "Throw std::inv_arg", 
                       pass ? "Thrown" : "No Exception", pass);
    }
    {
        total_checks++;
        bool pass = false;
        try {
            Tensor matmul_a({2, 3});
            Tensor matmul_b({4, 5});
            matmul_a.matmul(matmul_b);
        } catch (const std::invalid_argument&) {
            pass = true;
        }
        if (pass) passed_checks++;
        print_test_row("TC-18", "Edge Case: Inner dim mismatch", "Throw std::inv_arg", 
                       pass ? "Thrown" : "No Exception", pass);
    }

    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
    std::cout << "================================================================================" << std::endl;

    // Benchmark 1: GQA-sized math
    {
        std::vector<size_t> shapeA = {64, 16, 2048, 128};
        std::vector<size_t> shapeB = {64, 16, 128, 2048};
        Tensor MA(shapeA, 0.01f);
        Tensor MB(shapeB, 0.02f);
        
        std::cout << "GQA Matrix Multiplication:" << std::endl;
        std::cout << "  Dimensions:           [64, 16, 2048, 128] * [64, 16, 128, 2048]" << std::endl;
        double flops = 2.0 * 64.0 * 16.0 * 2048.0 * 128.0 * 2048.0;
        
        const int runs = 1;
        std::vector<double> latencies;
        latencies.reserve(runs);
        
        for (int r = 0; r < runs; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            Tensor MC = MA.matmul(MB);
            auto end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            latencies.push_back(duration_ms);
        }
        
        double avg_ms = latencies[0];
        double avg_seconds = avg_ms / 1000.0;
        double gflops = (flops / 1e9) / avg_seconds;
        
        std::cout << "  Average Latency:      " << avg_ms << " ms" << std::endl;
        std::cout << "  Throughput:           " << gflops << " GFLOPS" << std::endl << std::endl;
    }

    // Benchmark 2: Attention Projection (Width=2048)
    {
        Tensor MA({64, 2048, 2048}, 0.01f);
        Tensor MB({64, 2048, 2048}, 0.02f);
        
        std::cout << "Attention Projection:" << std::endl;
        std::cout << "  Dimensions:           [64, 2048, 2048] * [64, 2048, 2048]" << std::endl;
        double flops = 2.0 * 64.0 * 2048.0 * 2048.0 * 2048.0;
        
        const int runs = 1;
        std::vector<double> latencies;
        latencies.reserve(runs);
        
        for (int r = 0; r < runs; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            Tensor MC = MA.matmul(MB);
            auto end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            latencies.push_back(duration_ms);
        }
        
        double avg_ms = latencies[0];
        double avg_seconds = avg_ms / 1000.0;
        double gflops = (flops / 1e9) / avg_seconds;
        
        std::cout << "  Average Latency:      " << avg_ms << " ms" << std::endl;
        std::cout << "  Throughput:           " << gflops << " GFLOPS" << std::endl << std::endl;
    }

    // Benchmark 3: FFN Projection (Width=2048)
    {
        Tensor MA({64, 2048, 2048}, 0.01f);
        Tensor MB({64, 2048, 10922}, 0.02f);
        
        std::cout << "FFN Projection:" << std::endl;
        std::cout << "  Dimensions:           [64, 2048, 2048] * [64, 2048, 10922]" << std::endl;
        double flops = 2.0 * 64.0 * 2048.0 * 2048.0 * 10922.0;
        
        const int runs = 1;
        std::vector<double> latencies;
        latencies.reserve(runs);
        
        for (int r = 0; r < runs; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            Tensor MC = MA.matmul(MB);
            auto end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            latencies.push_back(duration_ms);
        }
        
        double avg_ms = latencies[0];
        double avg_seconds = avg_ms / 1000.0;
        double gflops = (flops / 1e9) / avg_seconds;
        
        std::cout << "  Average Latency:      " << avg_ms << " ms" << std::endl;
        std::cout << "  Throughput:           " << gflops << " GFLOPS" << std::endl;
    }

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
