#pragma once
#include <cstddef>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdio>

struct GemmRecord {
    size_t M, N, K;
    bool transA, transB;
    double elapsed_ms;
};

class GemmProfiler {
public:
    static GemmProfiler &instance() {
        static GemmProfiler p;
        return p;
    }

    void begin_step() { records_.clear(); }

    void record(size_t M, size_t N, size_t K, bool tA, bool tB, double ms) {
        records_.push_back({M, N, K, tA, tB, ms});
    }

    void print_summary(double step_gpu_ms) {
        struct Agg { double total_ms = 0; int count = 0; };
        std::unordered_map<std::string, Agg> agg;

        for (auto &r : records_) {
            char key[80];
            snprintf(key, 80, "%zu_%zu_%zu_%d_%d", r.M, r.N, r.K, r.transA, r.transB);
            agg[key].total_ms += r.elapsed_ms;
            agg[key].count++;
        }

        // Operation name mapped from dimensions + transpose flags
        auto op_name = [](size_t M, size_t N, size_t K, bool tA, bool tB) -> const char* {
            // Production shapes: B=32,S=1024 → BS=32768, H=1024, I=2752, V=100352
            if (!tA && !tB) {
                if (M == 32768 && K == 1024) {
                    if (N == 1024)  return "QKV_fwd_or_Wo_fwd";
                    if (N == 2752)  return "gate/up_fwd";
                    if (N == 100352) return "output_proj_fwd";
                }
                if (M == 1024 && K == 32768) {
                    if (N == 1024)  return "Wq/Wo_grad_bwd";
                    if (N == 2752)  return "Wgate/up_grad_bwd";
                }
                if (M == 2752 && K == 32768 && N == 1024) return "Wdown_grad_bwd";
            }
            if (tB && !tA) { // A @ B^T
                if (M == 32768) {
                    if (K == 1024 && N == 1024) return "grad_input_bwd (projT)";
                    if (K == 2752 && N == 1024) return "grad_ffnin_bwd (projT)";
                    if (K == 1024 && N == 2752) return "grad_ffnin_bwd (projT_up)";
                    if (K == 100352) return "grad_output_bwd (projT)";
                }
            }
            if (tA && !tB) { // A^T @ B
                if (M == 1024 && N == 1024 && K == 32768) return "dWq/Wo_bwd";
                if (M == 1024 && N == 2752 && K == 32768) return "dWgate/up_bwd";
                if (M == 1024 && N == 100352 && K == 32768) return "dWout_bwd";
                if (M == 2752 && N == 1024 && K == 32768) return "dWdown_bwd";
            }
            return "other";
        };

        // Convert to sortable
        std::vector<std::pair<std::string, Agg>> sorted(agg.begin(), agg.end());
        std::sort(sorted.begin(), sorted.end(),
            [](auto &a, auto &b) { return a.second.total_ms > b.second.total_ms; });

        printf("\n══════════════════════════════════════════════════════════════════════════════\n");
        printf("  GEMM PROFILER  (GPU time for this step: %.0f ms)\n", step_gpu_ms);
        printf("══════════════════════════════════════════════════════════════════════════════\n");
        printf("%-32s %4s %5s %5s  %5s %7s %7s\n",
               "Operation", "M", "N", "K", "Calls", "Enc(ms)", "Est.GFLOPS");
        printf("──────────────────────────────────────────────────────────────────────────────\n");

        // Estimate GFLOPS based on total GPU time (encoding time is tiny)
        // We compute per-call GFLOPS using: GFLOPS = 2*M*N*K / (GPU_time_per_call)
        double encode_total = 0;
        for (auto &[k_str, a] : sorted) encode_total += a.total_ms;

        for (auto &[key_str, a] : sorted) {
            size_t M,N,K; int tAi, tBi;
            sscanf(key_str.c_str(), "%zu_%zu_%zu_%d_%d", &M, &N, &K, &tAi, &tBi);
            auto name = op_name(M, N, K, tAi, tBi);
            double flops_total = 2.0 * static_cast<double>(M) * N * K * a.count;
            // Estimate GPU time = call_fraction * step_gpu_ms
            double call_fraction = a.count / (double)records_.size();
            double est_gpu_ms = call_fraction * step_gpu_ms;
            double est_gflops = flops_total / (est_gpu_ms / 1000.0) / 1e9;
            printf("%-32s %4zu %5zu %5zu  %5d %7.2f %7.0f\n",
                   name, M, N, K, a.count, a.total_ms, est_gflops);
        }
        printf("──────────────────────────────────────────────────────────────────────────────\n");
        int total_calls = (int)records_.size();
        printf("%-32s %4s %5s %5s  %5d %7.2f\n",
               "TOTAL", "", "", "", total_calls, encode_total);
        printf("══════════════════════════════════════════════════════════════════════════════\n");
        printf("  NOTE: 'Enc(ms)' = CPU encoding overhead only. GPU exec time estimated from\n");
        printf("  total step GPU time weighted by call count. True per-kernel GPU timing\n");
        printf("  requires MTLCommandBuffer instrumentation (not yet available per-kernel).\n");
        printf("══════════════════════════════════════════════════════════════════════════════\n\n");
        fflush(stdout);
    }

private:
    std::vector<GemmRecord> records_;
};
