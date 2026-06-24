#include "Tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdexcept>

// Helper to check if two floats are approximately equal
bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

void test_construction() {
    std::cout << "Running test_construction..." << std::endl;

    // 1. Default constructor
    Tensor t0;
    assert(t0.shape().empty());
    assert(t0.size() == 0 || t0.size() == 1);

    // 2. Shape constructor
    Tensor t1({2, 3});
    assert(t1.shape() == std::vector<size_t>({2, 3}));
    assert(t1.size() == 6);
    assert(t1.strides() == std::vector<size_t>({3, 1}));
    for (size_t i = 0; i < t1.size(); ++i) {
        assert(t1.data()[i] == 0.0f);
    }

    // 3. Fill constructor
    Tensor t2({2, 2}, 5.5f);
    assert(t2.shape() == std::vector<size_t>({2, 2}));
    assert(t2.size() == 4);
    assert(t2.strides() == std::vector<size_t>({2, 1}));
    for (size_t i = 0; i < t2.size(); ++i) {
        assert(t2.data()[i] == 5.5f);
    }

    // 4. Data constructor
    std::vector<float> custom_data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t3({2, 2}, custom_data);
    assert(t3.shape() == std::vector<size_t>({2, 2}));
    assert(t3.size() == 4);
    assert(t3.data() == custom_data);

    std::cout << "✓ test_construction passed!" << std::endl;
}

void test_indexing() {
    std::cout << "Running test_indexing..." << std::endl;

    Tensor t({2, 3, 4});
    
    // Fill with values
    float val = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                t(i, j, k) = val;
                val += 1.0f;
            }
        }
    }

    // Verify values
    val = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                assert(approx_equal(t(i, j, k), val));
                assert(approx_equal(t({i, j, k}), val));
                val += 1.0f;
            }
        }
    }

    std::cout << "✓ test_indexing passed!" << std::endl;
}

void test_elementwise() {
    std::cout << "Running test_elementwise..." << std::endl;

    Tensor A({2, 2}, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    Tensor B({2, 2}, std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}));

    // Out-of-place Add
    Tensor C = A.add(B);
    assert(C.data() == std::vector<float>({6.0f, 8.0f, 10.0f, 12.0f}));

    // In-place Add
    Tensor A_copy = A;
    A_copy.add_(B);
    assert(A_copy.data() == std::vector<float>({6.0f, 8.0f, 10.0f, 12.0f}));

    // Out-of-place Mul
    Tensor D = A.mul(B);
    assert(D.data() == std::vector<float>({5.0f, 12.0f, 21.0f, 32.0f}));

    // Out-of-place Scale
    Tensor E = A.scale(2.5f);
    assert(E.data() == std::vector<float>({2.5f, 5.0f, 7.5f, 10.0f}));

    std::cout << "✓ test_elementwise passed!" << std::endl;
}

void test_matmul() {
    std::cout << "Running test_matmul..." << std::endl;

    // A shape: [2, 3]
    // B shape: [3, 2]
    Tensor A({2, 3}, std::vector<float>({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }));
    Tensor B({3, 2}, std::vector<float>({
        7.0f,  8.0f,
        9.0f,  10.0f,
        11.0f, 12.0f
    }));

    Tensor C = A.matmul(B);

    assert(C.shape() == std::vector<size_t>({2, 2}));
    assert(approx_equal(C(0, 0), 58.0f));
    assert(approx_equal(C(0, 1), 64.0f));
    assert(approx_equal(C(1, 0), 139.0f));
    assert(approx_equal(C(1, 1), 154.0f));

    // Batched 3D Matrix multiplication
    Tensor A_batched({2, 2, 3}, std::vector<float>({
        // Batch 0
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        // Batch 1
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f
    }));

    Tensor B_batched({2, 3, 2}, std::vector<float>({
        // Batch 0
        7.0f,  8.0f,
        9.0f,  10.0f,
        11.0f, 12.0f,
        // Batch 1
        1.0f,  2.0f,
        3.0f,  4.0f,
        5.0f,  6.0f
    }));

    Tensor C_batched = A_batched.matmul(B_batched);
    assert(C_batched.shape() == std::vector<size_t>({2, 2, 2}));
    
    // Batch 0 verification
    assert(approx_equal(C_batched(0, 0, 0), 58.0f));
    assert(approx_equal(C_batched(0, 0, 1), 64.0f));
    assert(approx_equal(C_batched(0, 1, 0), 139.0f));
    assert(approx_equal(C_batched(0, 1, 1), 154.0f));

    // Batch 1 verification
    assert(approx_equal(C_batched(1, 0, 0), 9.0f));
    assert(approx_equal(C_batched(1, 0, 1), 12.0f));
    assert(approx_equal(C_batched(1, 1, 0), 18.0f));
    assert(approx_equal(C_batched(1, 1, 1), 24.0f));

    std::cout << "✓ test_matmul passed!" << std::endl;
}

void test_edge_cases() {
    std::cout << "Running test_edge_cases..." << std::endl;

    Tensor A({2, 3}, 1.0f);
    Tensor B({2, 3}, 2.0f);

    // 1. Index out of bounds (generic)
    try {
        A({2, 0});
        assert(false);
    } catch (const std::out_of_range& e) {
        // Expected
    }

    // 2. Index dimensionality mismatch
    try {
        A({0, 0, 0});
        assert(false);
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // 3. Shape mismatch in element-wise math
    Tensor C({2, 4}, 1.0f);
    try {
        A.add(C);
        assert(false);
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // 4. Batch mismatch in matmul
    Tensor A_batched({2, 2, 3}, 1.0f);
    Tensor B_batched({3, 3, 2}, 1.0f); // Batch 3 vs 2
    try {
        A_batched.matmul(B_batched);
        assert(false);
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // 5. Inner dimension mismatch in matmul
    Tensor D({2, 3}, 1.0f);
    Tensor E({4, 5}, 1.0f); // 3 cols of D != 4 rows of E
    try {
        D.matmul(E);
        assert(false);
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "✓ test_edge_cases passed!" << std::endl;
}

void test_performance() {
    std::cout << "Running test_performance (GQA-sized math)..." << std::endl;

    // Simulate GQA-like attention projection matrices
    // Query shape: [1, 8, 1024, 64] (Batch=1, Heads=8, SeqLen=1024, HeadDim=64)
    // Key shape (transposed): [1, 8, 64, 1024]
    std::vector<size_t> shapeA = {1, 8, 1024, 64};
    std::vector<size_t> shapeB = {1, 8, 64, 1024};

    std::cout << "  Initializing large matrices..." << std::endl;
    Tensor A(shapeA, 0.01f);
    Tensor B(shapeB, 0.02f);

    std::cout << "  Running matmul [1, 8, 1024, 64] * [1, 8, 64, 1024]..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor C = A.matmul(B);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Check shape is [1, 8, 1024, 1024]
    assert(C.shape() == std::vector<size_t>({1, 8, 1024, 1024}));
    
    // Check a value to verify math correctness
    // Each row of A has 64 elements of 0.01
    // Each col of B has 64 elements of 0.02
    // Dot product should be 64 * 0.01 * 0.02 = 64 * 0.0002 = 0.0128
    assert(approx_equal(C(0, 0, 0, 0), 0.0128f));

    // Compute FLOPs: 2 * num_batches * M * K * N
    // 2 * 8 * 1024 * 64 * 1024 = 1,073,741,824 FLOPs
    double flops = 2.0 * 8.0 * 1024.0 * 64.0 * 1024.0;
    double seconds = duration / 1000.0;
    double gflops = (flops / 1e9) / seconds;

    std::cout << "  Time taken: " << duration << " ms" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "✓ test_performance passed!" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "       Tensor Unit Test Runner" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    test_construction();
    test_indexing();
    test_elementwise();
    test_matmul();
    test_edge_cases();
    test_performance();

    std::cout << std::endl << "========================================" << std::endl;
    std::cout << "        All tests passed! 🎉" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
