#include "gpu_kernel/MetalBridge.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
int main() {

  metal_bridge::initialize();

  size_t size = 2 << 23;

  std::vector<float> a(size);
  std::vector<float> b(size);
  std::vector<float> c_cpu(size);
  std::vector<float> c_gpu(size);

  std::cout << "size: " << size << std::endl;

  int counter = 0;
  float scale = 1.12f;

  std::generate(a.begin(), a.end(),
                [&counter, &scale]() { return (counter++ * scale); });

  counter = 0;
  std::generate(b.begin(), b.end(),
                [&counter, scale]() { return (counter++ * scale * 1.2f); });

  auto cpu_start = std::chrono::high_resolution_clock::now();
  std::transform(a.begin(), a.end(), b.begin(), c_cpu.begin(),
                 [](const float a, const float b) { return a + b; });
  auto cpu_end = std::chrono::high_resolution_clock::now();

  auto gpu_start = std::chrono::high_resolution_clock::now();
  metal_bridge::vector_add(a.data(), b.data(), c_gpu.data(), size);
  auto gpu_end = std::chrono::high_resolution_clock::now();

  double cpu_duration_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start)
          .count() /
      1000.0;
  double gpu_duration_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start)
          .count() /
      1000.0;

  std::cout << "CPU duration: " << cpu_duration_ms << " ms" << std::endl;
  std::cout << "GPU duration: " << gpu_duration_ms << " ms" << std::endl;
  bool match = true;
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(c_cpu[i] - c_gpu[i]) > 1e-5) {
      match = false;
      break;
    }
  }
  std::cout << "Vector Add Verification: " << (match ? "PASS ✅" : "FAIL ❌")
            << std::endl;

  // ==============================================================================
  // MATRIX MULTIPLICATION (GEMM) BENCHMARK & VERIFICATION
  // ==============================================================================
  std::cout << "\n--- Commencing Matrix Multiplication (GEMM) Test ---" << std::endl;

  // Let's use shapes that align with our 128x128 tile size
  size_t gemm_M = 128;
  size_t gemm_K = 256;
  size_t gemm_N = 256;
  std::cout << "GEMM Shape: A[" << gemm_M << " x " << gemm_K << "] * B[" 
            << gemm_K << " x " << gemm_N << "] = C[" << gemm_M << " x " << gemm_N << "]" << std::endl;

  std::vector<float> gemm_A(gemm_M * gemm_K);
  std::vector<float> gemm_B(gemm_K * gemm_N);
  std::vector<float> gemm_C_cpu(gemm_M * gemm_N, 0.0f);
  std::vector<float> gemm_C_gpu(gemm_M * gemm_N, 0.0f);

  // Initialize inputs with deterministic scale values
  float val = 0.01f;
  for (size_t i = 0; i < gemm_A.size(); ++i) gemm_A[i] = (i % 7) * val;
  for (size_t i = 0; i < gemm_B.size(); ++i) gemm_B[i] = (i % 11) * val;

  // 1. CPU Reference Path (Simple 3-loop GEMM)
  auto gemm_cpu_start = std::chrono::high_resolution_clock::now();
  for (size_t r = 0; r < gemm_M; ++r) {
    for (size_t c = 0; c < gemm_N; ++c) {
      float sum = 0.0f;
      for (size_t k = 0; k < gemm_K; ++k) {
        sum += gemm_A[r * gemm_K + k] * gemm_B[k * gemm_N + c];
      }
      gemm_C_cpu[r * gemm_N + c] = sum;
    }
  }
  auto gemm_cpu_end = std::chrono::high_resolution_clock::now();

  // 2. GPU Custom Path (gemm_ffn)
  auto gemm_gpu_start = std::chrono::high_resolution_clock::now();
  metal_bridge::gemm_ffn(gemm_A.data(), gemm_B.data(), gemm_C_gpu.data(), gemm_M, gemm_N, gemm_K);
  auto gemm_gpu_end = std::chrono::high_resolution_clock::now();

  double gemm_cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(gemm_cpu_end - gemm_cpu_start).count() / 1000.0;
  double gemm_gpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(gemm_gpu_end - gemm_gpu_start).count() / 1000.0;

  std::cout << "GEMM CPU duration: " << gemm_cpu_ms << " ms" << std::endl;
  std::cout << "GEMM GPU duration: " << gemm_gpu_ms << " ms" << std::endl;

  // 3. Verification check
  bool gemm_match = true;
  for (size_t i = 0; i < gemm_C_cpu.size(); ++i) {
    if (std::abs(gemm_C_cpu[i] - gemm_C_gpu[i]) > 1e-4) {
      gemm_match = false;
      std::cout << "Mismatch at index " << i << " -> CPU: " << gemm_C_cpu[i] 
                << " GPU: " << gemm_C_gpu[i] << std::endl;
      break;
    }
  }
  std::cout << "GEMM GPU Verification: " << (gemm_match ? "PASS ✅" : "FAIL ❌") << std::endl;

  return 0;
}