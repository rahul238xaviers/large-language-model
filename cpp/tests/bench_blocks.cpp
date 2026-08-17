/**
 * @file bench_blocks.cpp
 * @brief Per-block instrumentation harness for GPU kernel performance analysis.
 *
 * Measures each pipeline stage in isolation with production shapes (380M model):
 *   batch=32, seq=1024, hidden=1024, intermediate=2752
 *   n_heads=16, n_kv_heads=8, head_dim=64, vocab=100352
 *
 * Reports: GPU kernel time, data transfer time, total wall time, GFLOPS, BW.
 *
 * Build:   make -C build bench_blocks && ./build/bench_blocks
 */

#include "Tensor.hpp"
#include "Attention.hpp"
#include "RMSNorm.hpp"
#include "Activations.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Transformer.hpp"
#include "TransformerConfig.hpp"
#include "gpu_kernel/MetalBridge.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <cmath>

// ── Production config (380M) ──────────────────────────────────────────────
static constexpr size_t B = 32, S = 1024, H = 1024, I = 2752;
static constexpr size_t nH = 16, nKV = 8, HD = 64, V = 100352;
static constexpr size_t M = B * S; // 32768

// Number of warmup + sample iterations (reduce for heavy kernels)
static constexpr int W = 2;   // warmup
static constexpr int SAMP = 3; // samples for element-wise
static constexpr int SAMP_GEMM = 3; // samples for GEMMs
static constexpr int SAMP_HEAVY = 2; // samples for heavy GEMMs (large K)

// ── Kernel call frequencies per training step (24-layer 1.6B model) ──────
static constexpr int FREQ_EMBEDDING     = 1;    // embedding_forward
static constexpr int FREQ_OUTPUT_PROJ   = 1;    // output projection forward
static constexpr int FREQ_FINAL_NORM    = 1;    // final rms_norm_forward
static constexpr int FREQ_PER_LAYER     = 24;
static constexpr int FREQ_RMS_FWD       = 48;   // 2 per layer × 24
static constexpr int FREQ_RMS_BWD       = 48;   // 2 per layer × 24
static constexpr int FREQ_ATTN_FUSED_BWD= 24;   // 1 per layer
static constexpr int FREQ_ROPE_FWD      = 24;   // 1 per layer
static constexpr int FREQ_ROPE_BWD      = 24;   // 1 per layer
static constexpr int FREQ_SWIGLU_BWD    = 24;   // 1 per layer
static constexpr int FREQ_DOWN_PROJ     = 24;   // 1 per layer
static constexpr int FREQ_RESIDUAL_ADD  = 72;   // 3 per layer × 24 (2 fwd + 1 bwd)
static constexpr int FREQ_ADAMW_PARAM   = 122;  // ~122 weight tensors
// backward GEMMs per layer
static constexpr int FREQ_GEMM_BWD_ATTN = 24;   // QKV weight grad (w_qkv fused)
static constexpr int FREQ_GEMM_BWD_GATE = 24;   // w_gate grad
static constexpr int FREQ_GEMM_BWD_UP   = 24;   // w_up grad
static constexpr int FREQ_GEMM_BWD_DOWN = 24;   // w_down grad
static constexpr int FREQ_GEMM_BWD_OUT  = 1;    // output_proj grad
static constexpr int FREQ_GEMM_BWD_EMB  = 1;    // embedding grad
static constexpr int FREQ_GEMM_PROJT_INPUT = 24;  // grad_input for attn (Wo^T)
static constexpr int FREQ_GEMM_PROJT_DOWN  = 24;  // grad_input for down_proj
static constexpr int FREQ_GEMM_PROJT_OUT   = 1;   // grad_input for output_proj
static constexpr int FREQ_CROSS_ENTROPY = 1;

// ── helpers ───────────────────────────────────────────────────────────────
static void set_gpu(bool on) {
    setenv("GPU_ENABLED", on ? "1" : "0", 1);
}

struct LedgerEntry {
    std::string name;
    double ms_per_call;       // measured GPU time for one call
    int calls_per_step;       // how many times per step
    double total_ms;          // ms_per_call × calls_per_step
    double bw_gbs;            // memory bandwidth
};

struct BenchResult {
    std::string label;
    double gpu_ms;    // GPU kernel execution time (from Metal timing)
    double copy_ms;   // data upload + copy-back time (estimated)
    double wall_ms;   // wall clock time
    double gflops;    // estimated GFLOPs
    double bw_gbs;    // estimated memory bandwidth GB/s
};

static double now_ms() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count() / 1000.0;
}

static void warmup_gpu() {
    set_gpu(true);
    metal_bridge::reconcile_buffers();
    size_t Mw = 128, Kw = 256, Nw = 256;
    std::vector<float> a(Mw * Kw, 0.01f), b(Kw * Nw, 0.01f), c(Mw * Nw, 0.0f);
    metal_bridge::gemm_bf16(a.data(), b.data(), c.data(), Mw, Nw, Kw);
}

// ── Per-block benchmarks ──────────────────────────────────────────────────

// 1. embedding forward: tokens[B,S] → h[B,S,H]
static BenchResult bench_embedding() {
    BenchResult r{"embedding_forward"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<uint32_t> tokens(M);
    for (auto &v : tokens) v = rand() % V;
    std::vector<float> table(V * H, 0.02f);
    std::vector<float> h(M * H, 0.0f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::embedding_forward(tokens.data(), table.data(), h.data(), M, H, V);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::embedding_forward(tokens.data(), table.data(), h.data(), M, H, V);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double bytes = (double)(M * H * 4) * 2; // read tokens + write h
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 2. rms_norm forward: [B,S,H] → [B,S,H]
static BenchResult bench_rmsnorm_fwd() {
    BenchResult r{"rms_norm_forward"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> input(M * H, 0.5f);
    std::vector<float> output(M * H, 0.0f);
    std::vector<float> weight(H, 1.0f);
    float eps = 1e-5f;

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::rms_norm_forward(input.data(), output.data(), weight.data(), eps, M, H);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::rms_norm_forward(input.data(), output.data(), weight.data(), eps, M, H);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double bytes = (double)(M * H * 4) * 3; // read input + weight, write output
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 3. rms_norm backward
static BenchResult bench_rmsnorm_bwd() {
    BenchResult r{"rms_norm_backward"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> grad_out(M * H, 0.5f);
    std::vector<float> input(M * H, 0.5f);
    std::vector<float> weight(H, 1.0f);
    std::vector<float> grad_in(M * H, 0.0f);
    std::vector<float> grad_w(H, 0.0f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::rms_norm_backward(grad_out.data(), input.data(), weight.data(),
                                         grad_in.data(), grad_w.data(), 1e-5f, M, H);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::rms_norm_backward(grad_out.data(), input.data(), weight.data(),
                                         grad_in.data(), grad_w.data(), 1e-5f, M, H);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double bytes = (double)(M * H * 4) * 5;
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// ── BF16 pack helper ────────────────────────────────────────────────
// Return a float whose lower and upper 16 bits hold the same bf16(val),
// so the GPU kernel sees val when reading 2 bytes at a time.
static float bf16_packed(float val) {
    uint32_t f32;
    memcpy(&f32, &val, 4);
    uint16_t bf16 = (uint16_t)(f32 >> 16);   // round-to-nearest-even via truncation
    uint32_t packed = ((uint32_t)bf16 << 16) | bf16;
    float result;
    memcpy(&result, &packed, 4);
    return result;
}
// Fill a float vector with bf16-packed copies of val (each float → 2 bf16 slots)
static std::vector<float> make_bf16_vec(size_t bf16_elems, float val) {
    size_t n_floats = (bf16_elems + 1) / 2;  // 2 bf16 per float
    float f = bf16_packed(val);
    std::vector<float> v(n_floats);
    std::fill(v.begin(), v.end(), f);
    return v;
}

// 4c. gemm_proj down GPU: backward-path GEMM with trB=true on small M.
//     C[M×N] = A[M×K] × B_logical[K×N] where B is stored as N×K.
//     Uses gemm_bf16_down pipeline (BM=32, coalesced B-load).
static BenchResult bench_gemm_proj_down_gpu() {
    BenchResult r{"gemm_proj down (GPU)"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    // Down-projection backward shape: M=32, N=2752, K=1024, trB=true
    size_t M = 32, N = 2752, K = 1024;

    auto A = make_bf16_vec(M * K, 0.5f);
    auto B = make_bf16_vec(N * K, 0.5f);   // stored as N×K
    auto C = make_bf16_vec(M * N, 0.0f);

    int warmup = W, samples = SAMP_GEMM;
    for (int i = 0; i < warmup; i++)
        metal_bridge::gemm_proj_down(A.data(), B.data(), C.data(), M, N, K);
    double t0 = now_ms();
    for (int i = 0; i < samples; i++)
        metal_bridge::gemm_proj_down(A.data(), B.data(), C.data(), M, N, K);
    r.wall_ms = (now_ms() - t0) / samples;

    double flops = 2.0 * M * K * N;
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;

    // Bandwidth uses the buffer bytes actually transferred (bf16: 2 bytes/elem)
    double bytes = (double)((M * K + N * K + M * N) * 2);
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 4d. gemm_bwd_weight GPU: coalesced weight-gradient GEMM (trA=true, K=32).
static BenchResult bench_gemm_bwd_weight_gpu(size_t M_, size_t N_,
                                             const std::string &tag) {
    BenchResult r{"gemm_bwd " + tag};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    size_t K_ = 32;
    auto A = make_bf16_vec(K_ * M_, 0.5f);   // stored as K×M
    auto B = make_bf16_vec(K_ * N_, 0.5f);   // K×N
    auto C = make_bf16_vec(M_ * N_, 0.0f);

    int warmup = W, samples = SAMP_HEAVY;
    for (int i = 0; i < warmup; i++)
        metal_bridge::gemm_bwd_weight(A.data(), B.data(), C.data(), M_, N_, K_);
    double t0 = now_ms();
    for (int i = 0; i < samples; i++)
        metal_bridge::gemm_bwd_weight(A.data(), B.data(), C.data(), M_, N_, K_);
    r.wall_ms = (now_ms() - t0) / samples;

    double flops = 2.0 * M_ * K_ * N_;
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;

    double bytes = (double)((K_ * M_ + K_ * N_ + M_ * N_) * 2);
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 5. gemm_backward: C += A^T @ B  (weight gradient)
//    A is stored as [K×M] in memory (transposed), B as [K×N], C as [M×N]
static BenchResult bench_gemm_bwd(size_t M_, size_t N_, size_t K_,
                                  const std::string &tag) {
    BenchResult r{"gemm_bwd " + tag};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    // A stored as K×M in memory (the "transposed" layout)
    std::vector<float> A(K_ * M_, 0.5f);
    std::vector<float> B(K_ * N_, 0.5f);
    std::vector<float> C(M_ * N_, 0.0f);

    int warmup = W, samples = SAMP_HEAVY;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::gemm_backward(A.data(), B.data(), C.data(), M_, N_, K_);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::gemm_backward(A.data(), B.data(), C.data(), M_, N_, K_);
    }
    r.wall_ms = (now_ms() - t0) / samples;

    double flops = 2.0 * M_ * N_ * K_;
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 6. gemm_proj_trans_b: C = A @ B^T  (input gradient)
//    A [M×K], B stored as [N×K] in memory
static BenchResult bench_gemm_proj_trans_b(size_t M_, size_t N_, size_t K_,
                                           const std::string &tag) {
    BenchResult r{"gemm_projT " + tag};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> A(M_ * K_, 0.5f);
    std::vector<float> B(N_ * K_, 0.5f);
    std::vector<float> C(M_ * N_, 0.0f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::gemm_proj_trans_b(A.data(), B.data(), C.data(), M_, N_, K_);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::gemm_proj_trans_b(A.data(), B.data(), C.data(), M_, N_, K_);
    }
    r.wall_ms = (now_ms() - t0) / samples;

    double flops = 2.0 * M_ * N_ * K_;
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 7. gemm_gqa: fused GQA attention
static BenchResult bench_gqa() {
    BenchResult r{"gemm_gqa"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> Q(B * nH * S * HD, 0.1f);
    std::vector<float> K(B * nKV * S * HD, 0.1f);
    std::vector<float> V(B * nKV * S * HD, 0.1f);
    std::vector<float> out(B * S * nH * HD, 0.0f);

    metal_bridge::GQAParams p;
    p.batch = B; p.n_q_heads = nH; p.n_kv_heads = nKV;
    p.seq_len = S; p.head_dim = HD;

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::gemm_gqa(p, Q.data(), K.data(), V.data(), out.data());
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::gemm_gqa(p, Q.data(), K.data(), V.data(), out.data());
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double flops = 2.0 * B * nH * S * S * HD; // QK^T + PV softmax
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 8. cross_entropy
static BenchResult bench_cross_entropy() {
    BenchResult r{"cross_entropy"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> logits(M * V, 0.1f);
    std::vector<uint32_t> targets(M);
    for (auto &v : targets) v = rand() % V;
    std::vector<float> grad(M * V, 0.0f);
    float loss = 0.0f;

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::cross_entropy(logits.data(), targets.data(), &loss, grad.data(), M, V);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::cross_entropy(logits.data(), targets.data(), &loss, grad.data(), M, V);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double bytes = (double)(M * V * 4 + M * 4 + M * V * 4 + 4);
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 10. adamw_step (single param)
static BenchResult bench_adamw() {
    BenchResult r{"adamw_step (H×I)"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    size_t n = H * I;
    std::vector<float> param(n, 0.1f);
    std::vector<float> grad(n, 0.01f);
    std::vector<float> m(n, 0.0f);
    std::vector<float> v(n, 0.0f);
    metal_bridge::AdamWStepParams p;
    p.lr = 3e-4f; p.beta1 = 0.9f; p.beta2 = 0.999f; p.eps = 1e-8f;
    p.weight_decay = 0.1f; p.bias_correction1 = 0.9f; p.bias_correction2 = 0.999f;
    p.n = n;

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::adamw_step(param.data(), grad.data(), m.data(), v.data(), p);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::adamw_step(param.data(), grad.data(), m.data(), v.data(), p);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double flops = 6.0 * n;
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 11. residual_add
static BenchResult bench_residual_add() {
    BenchResult r{"residual_add (B,S,H)"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> a(M * H, 0.5f);
    std::vector<float> b(M * H, 0.3f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::residual_add(a.data(), b.data(), M * H);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::residual_add(a.data(), b.data(), M * H);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double bytes = (double)(M * H * 4) * 3;
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 12. swiglu_backward
static BenchResult bench_swiglu_bwd() {
    BenchResult r{"swiglu_bwd (B,S,I)"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    size_t n = M * I;
    std::vector<float> grad_out(n, 0.5f);
    std::vector<float> gate(n, 0.5f);
    std::vector<float> up(n, 0.5f);
    std::vector<float> grad_gate(n, 0.0f);
    std::vector<float> grad_up(n, 0.0f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::swiglu_backward(grad_out.data(), gate.data(), up.data(),
                                       grad_gate.data(), grad_up.data(), n);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::swiglu_backward(grad_out.data(), gate.data(), up.data(),
                                       grad_gate.data(), grad_up.data(), n);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double bytes = (double)(n * 4) * 5;
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}


// 14. fused_attn_bwd: single-kernel GQA attention backward
static BenchResult bench_fused_attn_bwd() {
    BenchResult r{"fused_attn_bwd"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    size_t bytesQ   = B * nH * S * HD * sizeof(float);
    size_t bytesKV  = B * nKV * S * HD * sizeof(float);
    std::vector<float> Q(B * nH * S * HD, 0.1f);
    std::vector<float> K(B * nKV * S * HD, 0.1f);
    std::vector<float> V(B * nKV * S * HD, 0.1f);
    std::vector<float> dO(B * nH * S * HD, 0.5f);
    std::vector<float> dQ(B * nH * S * HD, 0.0f);
    std::vector<float> dK(B * nKV * S * HD, 0.0f);
    std::vector<float> dV(B * nKV * S * HD, 0.0f);

    int warmup = W, samples = SAMP_HEAVY;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::begin_scope();
        metal_bridge::fused_attn_bwd(Q.data(), K.data(), V.data(), dO.data(),
                                     dQ.data(), dK.data(), dV.data(),
                                     B, nH, nKV, S, HD);
        metal_bridge::end_scope();
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::begin_scope();
        metal_bridge::fused_attn_bwd(Q.data(), K.data(), V.data(), dO.data(),
                                     dQ.data(), dK.data(), dV.data(),
                                     B, nH, nKV, S, HD);
        metal_bridge::end_scope();
    }
    r.wall_ms = (now_ms() - t0) / samples;
    r.gpu_ms = r.wall_ms;

    double total_bytes = double(bytesQ * 2 + bytesKV * 4); // read: Q,K,V,dO; write: dQ,dK,dV
    r.bw_gbs = total_bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 15. rope_forward
static BenchResult bench_rope_fwd() {
    BenchResult r{"rope_forward"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    size_t half_dim = HD / 2;
    std::vector<float> Q(B * nH * S * HD, 0.1f);
    std::vector<float> K(B * nKV * S * HD, 0.1f);
    std::vector<float> cos_t(S * half_dim, 0.5f);
    std::vector<float> sin_t(S * half_dim, 0.5f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::begin_scope();
        metal_bridge::rope_forward(Q.data(), K.data(), cos_t.data(), sin_t.data(),
                                   B, nH, nKV, S, HD);
        metal_bridge::end_scope();
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::begin_scope();
        metal_bridge::rope_forward(Q.data(), K.data(), cos_t.data(), sin_t.data(),
                                   B, nH, nKV, S, HD);
        metal_bridge::end_scope();
    }
    r.wall_ms = (now_ms() - t0) / samples;
    r.gpu_ms = r.wall_ms;
    double bytes = double(B * S * HD * 4) * 4; // Q+K+cos+sin read, Q+K write
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 16. rope_backward
static BenchResult bench_rope_bwd() {
    BenchResult r{"rope_backward"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> grad(B * nH * S * HD, 0.5f);
    std::vector<float> cos_t(S * HD / 2, 0.5f);
    std::vector<float> sin_t(S * HD / 2, 0.5f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::begin_scope();
        metal_bridge::rope_backward(grad.data(), cos_t.data(), sin_t.data(),
                                    B, nH, S, HD);
        metal_bridge::end_scope();
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::begin_scope();
        metal_bridge::rope_backward(grad.data(), cos_t.data(), sin_t.data(),
                                    B, nH, S, HD);
        metal_bridge::end_scope();
    }
    r.wall_ms = (now_ms() - t0) / samples;
    r.gpu_ms = r.wall_ms;
    double bytes = double(B * nH * S * HD * 4) * 3;
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 17. adamw_step for different size classes
static BenchResult bench_adamw_size(size_t n, const std::string &tag) {
    BenchResult r{"adamw_step " + tag};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> param(n, 0.1f);
    std::vector<float> grad(n, 0.01f);
    std::vector<float> m(n, 0.0f);
    std::vector<float> v(n, 0.0f);
    metal_bridge::AdamWStepParams p;
    p.lr = 3e-4f; p.beta1 = 0.9f; p.beta2 = 0.999f; p.eps = 1e-8f;
    p.weight_decay = 0.1f; p.bias_correction1 = 0.9f; p.bias_correction2 = 0.999f;
    p.n = n;

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::begin_scope();
        metal_bridge::adamw_step(param.data(), grad.data(), m.data(), v.data(), p);
        metal_bridge::end_scope();
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::begin_scope();
        metal_bridge::adamw_step(param.data(), grad.data(), m.data(), v.data(), p);
        metal_bridge::end_scope();
    }
    r.wall_ms = (now_ms() - t0) / samples;
    r.gpu_ms = r.wall_ms;
    double bytes = double(n * 4) * 4; // param+grad+m+v
    r.bw_gbs = bytes / (r.wall_ms / 1000.0) / 1e9;
    return r;
}



// ── Layer-level forward pass ──────────────────────────────────────────────
static BenchResult bench_layer_fwd() {
    BenchResult r{"one_layer_forward (full)"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    ModelConfig cfg;
    cfg.hidden_dim = H; cfg.intermediate_dim = I;
    cfg.n_heads = nH; cfg.n_kv_heads = nKV;
    cfg.head_dim = HD; cfg.max_seq_len = S;
    cfg.vocab_size = V; cfg.rms_norm_eps = 1e-5f;

    TransformerLayer layer(cfg);
    // fill weights
    layer.w_gate.fill(0.02f); layer.w_up.fill(0.02f); layer.w_down.fill(0.02f);
    layer.attn.Wq().fill(0.02f); layer.attn.Wk().fill(0.02f);
    layer.attn.Wv().fill(0.02f); layer.attn.Wo().fill(0.02f);

    RoPE rope(HD, S, 10000.0f);
    Tensor h({B, S, H}, 0.5f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::start_batch();
        Tensor attn_in = layer.attn_norm.forward(h);
        Tensor attn_out = layer.attn.forward(attn_in, rope);
        h.add_(attn_out);
        Tensor ffn_in = layer.ffn_norm.forward(h);
        Tensor gate_proj = ffn_in.matmul(layer.w_gate);
        Tensor up_proj = ffn_in.matmul(layer.w_up);
        Tensor activated = activatations::swiglu(gate_proj, up_proj);
        Tensor ffn_out = activated.matmul(layer.w_down);
        h.add_(ffn_out);
        metal_bridge::commit_batch();
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::start_batch();
        Tensor attn_in = layer.attn_norm.forward(h);
        Tensor attn_out = layer.attn.forward(attn_in, rope);
        h.add_(attn_out);
        Tensor ffn_in = layer.ffn_norm.forward(h);
        Tensor gate_proj = ffn_in.matmul(layer.w_gate);
        Tensor up_proj = ffn_in.matmul(layer.w_up);
        Tensor activated = activatations::swiglu(gate_proj, up_proj);
        Tensor ffn_out = activated.matmul(layer.w_down);
        h.add_(ffn_out);
        metal_bridge::commit_batch();
    }
    r.wall_ms = (now_ms() - t0) / samples;
    return r;
}

// ── Layer-level backward pass ─────────────────────────────────────────────
static BenchResult bench_layer_bwd() {
    BenchResult r{"one_layer_backward (full)"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    ModelConfig cfg;
    cfg.hidden_dim = H; cfg.intermediate_dim = I;
    cfg.n_heads = nH; cfg.n_kv_heads = nKV;
    cfg.head_dim = HD; cfg.max_seq_len = S;
    cfg.vocab_size = V; cfg.rms_norm_eps = 1e-5f;

    TransformerLayer layer(cfg);
    layer.w_gate.fill(0.02f); layer.w_up.fill(0.02f); layer.w_down.fill(0.02f);
    layer.attn.Wq().fill(0.02f); layer.attn.Wk().fill(0.02f);
    layer.attn.Wv().fill(0.02f); layer.attn.Wo().fill(0.02f);

    RoPE rope(HD, S, 10000.0f);
    Tensor grad_output({B, S, H}, 0.5f);
    Tensor h_in({B, S, H}, 0.5f);

    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    int warmup = W, samples = SAMP_HEAVY;
    for (int i = 0; i < warmup; i++) {
        grad_w_gate.fill(0.0f); grad_w_up.fill(0.0f); grad_w_down.fill(0.0f);
        grad_Wq.fill(0.0f); grad_Wk.fill(0.0f); grad_Wv.fill(0.0f); grad_Wo.fill(0.0f);
        layer.backward(grad_output, h_in,
                       grad_w_gate, grad_w_up, grad_w_down,
                       grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        grad_w_gate.fill(0.0f); grad_w_up.fill(0.0f); grad_w_down.fill(0.0f);
        grad_Wq.fill(0.0f); grad_Wk.fill(0.0f); grad_Wv.fill(0.0f); grad_Wo.fill(0.0f);
        layer.backward(grad_output, h_in,
                       grad_w_gate, grad_w_up, grad_w_down,
                       grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    return r;
}

// ── Data pipeline overhead ────────────────────────────────────────────────
static BenchResult bench_stepcache_overhead() {
    BenchResult r{"stepCache+memcpy overhead"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    // simulate what happens per step: reconcile, then allocate+upload+copyback
    size_t bytes = M * H * 4; // 128 MB
    std::vector<float> data(M * H, 0.5f);
    std::vector<float> result(M * H, 0.0f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::gemm_bf16(data.data(), data.data(), result.data(), M, H, H);
    }
    metal_bridge::reconcile_buffers();
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        // This forces a stepCache miss → upload + copy-back + alloc
        metal_bridge::gemm_bf16(data.data(), data.data(), result.data(), M, H, H);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    return r;
}

// ── Main ──────────────────────────────────────────────────────────────────
int main() {
    std::cout << "\n══════════════════════════════════════════════════════════════\n";
    std::cout << "  PER-BLOCK GPU BENCHMARK  (B=" << B << " S=" << S << " H=" << H << " I=" << I << ")\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    set_gpu(true);
    metal_bridge::initialize();
    warmup_gpu();

    std::vector<BenchResult> results;

    // ── Element-wise / lightweight ──
    results.push_back(bench_rmsnorm_fwd());
    results.push_back(bench_rmsnorm_bwd());
    results.push_back(bench_residual_add());
    results.push_back(bench_swiglu_bwd());
    results.push_back(bench_adamw());
    results.push_back(bench_embedding());
    results.push_back(bench_cross_entropy());
    results.push_back(bench_rope_fwd());
    results.push_back(bench_rope_bwd());

    // ── Attention-specific ──
    results.push_back(bench_fused_attn_bwd());

    // ── GEMMs ──
    results.push_back(bench_gqa());

    // ── Backward GEMMs (GPU, coalesced) ──
    results.push_back(bench_gemm_bwd_weight_gpu(H, H, "Wq/Wo grad (H,M)"));   // 1024×1024×32
    results.push_back(bench_gemm_bwd_weight_gpu(H, I, "Wgate/Wup grad (I,M)"));// 1024×2752×32
    results.push_back(bench_gemm_bwd_weight_gpu(I, H, "Wdown grad (I,M)"));    // 2752×1024×32
    results.push_back(bench_gemm_proj_down_gpu());                     // down: GPU (gemm_bf16_down)
    results.push_back(bench_gemm_proj_trans_b(M, H, H, "grad_input (M,H)")); // 32768×1024×1024
    results.push_back(bench_gemm_proj_trans_b(M, H, I, "grad_ffn_in (M,I)"));// 32768×1024×2752

    // ── AdamW for different parameter sizes ──
    results.push_back(bench_adamw_size(H * H, "(H×H=1M)"));      // attn weights
    results.push_back(bench_adamw_size(H * I, "(H×I=2.8M)"));   // FFN weights
    results.push_back(bench_adamw_size(V * H, "(V×H=103M)"));   // embedding/output_proj

    // ── Layer-level ──
    results.push_back(bench_layer_fwd());
    results.push_back(bench_layer_bwd());

    // ── Overhead ──
    results.push_back(bench_stepcache_overhead());

    // ── Print per-block table ──
    double total_fwd_bwd_ms = 0, total_fwd_bwd_target = 1350.0;
    for (auto &r : results) {
        if (r.label.find("one_layer") != std::string::npos || r.label == "stepCache+memcpy overhead")
            continue;
        total_fwd_bwd_ms += r.wall_ms;
    }

    std::cout << std::left << std::setw(30) << "Block"
              << std::setw(14) << "Wall (ms)"
              << std::setw(14) << "Target (ms)"
              << std::setw(14) << "GFLOPS"
              << std::setw(14) << "BW (GB/s)"
              << "\n";
    std::cout << std::string(86, '-') << "\n";
    for (auto &r : results) {
        double target = (r.label.find("one_layer") != std::string::npos ||
                         r.label == "stepCache+memcpy overhead")
                            ? 0
                            : r.wall_ms / total_fwd_bwd_ms * total_fwd_bwd_target;
        std::cout << std::left << std::setw(30) << r.label
                  << std::fixed << std::setprecision(2) << std::setw(14) << r.wall_ms
                  << std::fixed << std::setprecision(2) << std::setw(14) << target
                  << std::fixed << std::setprecision(1) << std::setw(14) << r.gflops
                  << std::fixed << std::setprecision(1) << std::setw(14) << r.bw_gbs
                  << "\n";
    }

    // ── Bottleneck Ledger ─────────────────────────────────────────────────
    std::cout << "\n══════════════════════════════════════════════════════════════════════\n";
    std::cout << "  BOTTLENECK LEDGER — Extrapolated Step Cost (24 layers, 1.6B params)\n";
    std::cout << "══════════════════════════════════════════════════════════════════════\n";
    std::cout << std::left << std::setw(30) << "Kernel"
              << std::right
              << std::setw(16) << "ms/call"
              << std::setw(14) << "Calls/step"
              << std::setw(16) << "Total (ms)"
              << std::setw(14) << "BW (GB/s)"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    // Build ledger entries from measured results
    auto find_ms = [&](const std::string &prefix) -> double {
        for (auto &r : results)
            if (r.label.find(prefix) == 0) return r.wall_ms;
        return 0;
    };

    std::vector<LedgerEntry> ledger;

    //embedding
    ledger.push_back({"embedding_forward", find_ms("embedding_forward"), FREQ_EMBEDDING, 0, 0});
    // rms_norm fwd (2 per layer)
    ledger.push_back({"rms_norm_forward", find_ms("rms_norm_forward"), FREQ_RMS_FWD, 0, 0});
    // rms_norm bwd (2 per layer)
    ledger.push_back({"rms_norm_backward", find_ms("rms_norm_backward"), FREQ_RMS_BWD, 0, 0});
    // rope forward
    ledger.push_back({"rope_forward", find_ms("rope_forward"), FREQ_ROPE_FWD, 0, 0});
    // rope backward
    ledger.push_back({"rope_backward", find_ms("rope_backward"), FREQ_ROPE_BWD, 0, 0});

    // gemm_gqa (attention output — PV)
    ledger.push_back({"gemm_gqa (attn_out)", find_ms("gemm_gqa"), FREQ_PER_LAYER, 0, 0});
    // fused_attn_bwd
    ledger.push_back({"fused_attn_bwd", find_ms("fused_attn_bwd"), FREQ_ATTN_FUSED_BWD, 0, 0});
    // swiglu backward
    ledger.push_back({"swiglu_backward", find_ms("swiglu_bwd"), FREQ_SWIGLU_BWD, 0, 0});
    // down proj backward gemm (GPU, gemm_bf16_down)
    ledger.push_back({"gemm_proj down (GPU)", find_ms("gemm_proj down (GPU)"), FREQ_PER_LAYER, 0, 0});
    // residual_add (3 per layer)
    ledger.push_back({"residual_add", find_ms("residual_add (B,S,H)"), FREQ_RESIDUAL_ADD, 0, 0});
    // backward GEMMs
    ledger.push_back({"gemm_bwd Wq/Wo", find_ms("gemm_bwd Wq/Wo grad (H,M)"), FREQ_GEMM_BWD_ATTN, 0, 0});
    ledger.push_back({"gemm_bwd Wgate/Wup", find_ms("gemm_bwd Wgate/Wup grad (I,M)"), FREQ_GEMM_BWD_GATE, 0, 0});
    ledger.push_back({"gemm_bwd Wdown", find_ms("gemm_bwd Wdown grad (I,M)"), FREQ_GEMM_BWD_DOWN, 0, 0});
    ledger.push_back({"gemm_projT grad_input", find_ms("gemm_projT grad_input (M,H)"), FREQ_GEMM_PROJT_INPUT, 0, 0});
    // cross_entropy
    ledger.push_back({"cross_entropy", find_ms("cross_entropy"), FREQ_CROSS_ENTROPY, 0, 0});

    // Optimizer — weighted by parameter count
    double adamw_HH = find_ms("adamw_step (H×H=1M)");
    double adamw_HI = find_ms("adamw_step (H×I=2.8M)");
    double adamw_VH = find_ms("adamw_step (V×H=103M)");
    // Count params per size class
    int n_attn_weights = 24 * 4; // Wq,WkV,Wv,Wo per layer — each H×H
    int n_ffn_weights  = 24 * 3; // Wgate,Wup,Wdown per layer — each H×I
    int n_emb_weights = 2;       // embedding + output_proj — each V×H
    double opt_ms_HH = adamw_HH * n_attn_weights;
    double opt_ms_HI = adamw_HI * n_ffn_weights;
    double opt_ms_VH = adamw_VH * n_emb_weights;
    // Per-size-class BW (GB/s): bytes = 4 params × 4 bytes × n
    double bw_HH = (H * H * 4.0 * 4) / (adamw_HH * 1e-3) / 1e9;
    double bw_HI = (H * I * 4.0 * 4) / (adamw_HI * 1e-3) / 1e9;
    double bw_VH = (V * H * 4.0 * 4) / (adamw_VH * 1e-3) / 1e9;
    ledger.push_back({"adamw_step (1024×1024,1M)", adamw_HH, n_attn_weights, 0, bw_HH});
    ledger.push_back({"adamw_step (1024×2752,2.8M)", adamw_HI, n_ffn_weights, 0, bw_HI});
    ledger.push_back({"adamw_step (100352×1024,103M)", adamw_VH, n_emb_weights, 0, bw_VH});

    // Compute totals and print
    double ledger_total = 0;
    for (auto &e : ledger) {
        e.total_ms = e.ms_per_call * e.calls_per_step;
        // Use passed-in BW from measurement if available, else compute
        for (auto &r : results) {
            if (r.label.find(e.name) == 0 && r.bw_gbs > 0) {
                e.bw_gbs = r.bw_gbs;
                break;
            }
        }
    }

    // Sort by total_ms descending
    std::sort(ledger.begin(), ledger.end(),
        [](const LedgerEntry &a, const LedgerEntry &b) { return a.total_ms > b.total_ms; });

    for (auto &e : ledger) {
        ledger_total += e.total_ms;
        std::cout << std::left << std::setw(30) << e.name
                  << std::right
                  << std::fixed << std::setprecision(3) << std::setw(16) << e.ms_per_call
                  << std::setw(14) << e.calls_per_step
                  << std::fixed << std::setprecision(1) << std::setw(16) << e.total_ms
                  << std::fixed << std::setprecision(1) << std::setw(14) << e.bw_gbs
                  << "\n";
    }
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left << std::setw(30) << "TOTAL"
              << std::right
              << std::setw(16) << ""
              << std::setw(14) << ""
              << std::fixed << std::setprecision(1) << std::setw(16) << ledger_total
              << std::setw(14) << ""
              << "\n";

    double step_budget = 32400.0; // 32.4s in ms
    double over_budget = ledger_total - step_budget;
    double pct_of_budget = ledger_total / step_budget * 100.0;
    std::cout << "\n  Budget:   " << step_budget / 1000.0 << " s/step\n";
    std::cout << "  Measured: " << ledger_total / 1000.0 << " s/step  (" << pct_of_budget << "% of budget)\n";
    if (over_budget > 0) {
        std::cout << "  OVER by:  " << over_budget / 1000.0 << " s\n";
    } else {
        std::cout << "  Under by: " << (-over_budget) / 1000.0 << " s  ✓\n";
    }

    // ── Top contributors to over-budget ──
    std::cout << "\n── Top 5 over-budget contributors ──\n";
    double accounted = 0;
    int rank = 0;
    for (auto &e : ledger) {
        if (rank >= 5) break;
        double fraction = e.total_ms / ledger_total * 100.0;
        std::cout << "  " << (rank+1) << ". " << std::left << std::setw(28) << e.name
                  << std::right << std::setw(8) << std::fixed << std::setprecision(1) << fraction << "%"
                  << "  (" << e.total_ms/1000.0 << " s)\n";
        accounted += e.total_ms;
        rank++;
    }
    std::cout << "  Top 5 total: " << accounted / 1000.0 << " s  ("
              << accounted / ledger_total * 100.0 << "% of total)\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    return 0;
}
