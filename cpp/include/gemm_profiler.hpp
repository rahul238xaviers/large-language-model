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

    void begin_step() { records_.clear(); step_start_ = now(); }

    void record(size_t M, size_t N, size_t K, bool tA, bool tB, double ms) {
        records_.push_back({M, N, K, tA, tB, ms});
    }

    void print_summary() {
        // Build string key for grouping
        struct Agg { double total_ms = 0; int count = 0; };
        std::unordered_map<std::string, Agg> agg;

        for (auto &r : records_) {
            char key[64];
            snprintf(key, 64, "%zu_%zu_%zu_%d_%d", r.M, r.N, r.K, r.transA, r.transB);
            agg[key].total_ms += r.elapsed_ms;
            agg[key].count++;
        }

        // Operation name lookup from dimensions
        auto op_name = [](size_t M, size_t N, size_t K, bool tA, bool tB) -> const char* {
            size_t BS = 32768, H = 1024, I = 2752, V = 100352;
            if (!tA && !tB) {
                if (N == H && K == H && M == BS) return "QKV_fwd";
                if (N == I && K == H && M == BS) return "gate/up_fwd";
                if (N == H && K == I && M == BS) return "down_fwd";
                if (N == V && K == H && M == BS) return "output_proj_fwd";
                if (M == H && K == BS) {
                    if (N == H) return "Wq/Wo_grad";
                    if (N == I) return "Wgate/up_grad";
                }
                if (M == I && N == H && K == BS) return "Wdown_grad";
            }
            if (tB && !tA) {
                if (N == H && K == H) return "grad_input (projT)";
                if (N == H && K == I) return "grad_ffnin (projT)";
                if (K == V) return "grad_output (projT)";
            }
            if (tA && !tB) {
                if (M == H) {
                    if (N == V) return "dWout";
                    if (N == I) return "dWgate/up";
                    if (N == H) return "dWq/Wo";
                }
                if (M == I && N == H) return "dWdown";
            }
            return "other";
        };

        // Convert to sortable vector
        std::vector<std::pair<std::string, Agg>> sorted(agg.begin(), agg.end());
        std::sort(sorted.begin(), sorted.end(),
            [](auto &a, auto &b) { return a.second.total_ms > b.second.total_ms; });

        double grand_total = 0;
        for (auto &[k, a] : sorted) grand_total += a.total_ms;

        printf("\n══════════════════════════════════════════════════════════════════════════\n");
        printf("  GEMM PROFILER (batch=32, seq=1024, step total=%.0f ms)\n", grand_total);
        printf("══════════════════════════════════════════════════════════════════════════\n");
        printf("%-24s %14s %5s %8s %8s %8s\n",
               "Operation", "Shape", "Calls", "Tot(ms)", "%Time", "GFLOPS");
        printf("──────────────────────────────────────────────────────────────────────────\n");

        for (auto &[key_str, a] : sorted) {
            size_t M,N,K; bool tA,tB;
            sscanf(key_str.c_str(), "%zu_%zu_%zu_%d_%d", &M, &N, &K, (int*)&tA, (int*)&tB);
            auto name = op_name(M, N, K, tA, tB);
            double flops = 2.0 * M * N * K * a.count;
            double gflops = flops / (a.total_ms / 1000.0) / 1e9;
            double pct = a.total_ms / grand_total * 100;
            printf("%-24s %4zux%-6zu %4d %8.1f %7.1f%% %8.1f\n",
                   name, M, N, a.count, a.total_ms, pct, gflops);
        }
        printf("──────────────────────────────────────────────────────────────────────────\n");
        printf("%-24s %14s %4d %8.1f %7.1f%% %8s\n",
               "TOTAL", "", (int)records_.size(), grand_total, 100.0, "");
        printf("══════════════════════════════════════════════════════════════════════════\n\n");
        fflush(stdout);
    }

private:
    double now() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count() / 1000.0;
    }
    std::vector<GemmRecord> records_;
    double step_start_ = 0;
};
