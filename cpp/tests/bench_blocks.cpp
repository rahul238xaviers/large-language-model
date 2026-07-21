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

// ── helpers ───────────────────────────────────────────────────────────────
static void set_gpu(bool on) {
    setenv("GPU_ENABLED", on ? "1" : "0", 1);
}

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
    metal_bridge::gemm_proj(a.data(), b.data(), c.data(), Mw, Nw, Kw);
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

// 4. gemm_proj: matmul [M,H] × [H,N] → [M,N]
static BenchResult bench_gemm_proj(size_t N, const std::string &tag) {
    BenchResult r{"gemm_proj " + tag};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> A(M * H, 0.5f);
    std::vector<float> B(H * N, 0.5f);
    std::vector<float> C(M * N, 0.0f);

    int warmup = W, samples = SAMP_GEMM;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::gemm_proj(A.data(), B.data(), C.data(), M, N, H);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::gemm_proj(A.data(), B.data(), C.data(), M, N, H);
    }
    r.wall_ms = (now_ms() - t0) / samples;

    double flops = 2.0 * M * H * N;
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;

    double bytes = (double)((M * H + H * N + M * N) * 4);
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

// 8. gemm_ffn: fused SwiGLU FFN
static BenchResult bench_gemm_ffn() {
    BenchResult r{"gemm_ffn"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> A(M * H, 0.5f);
    std::vector<float> gate(H * I, 0.5f);
    std::vector<float> up(H * I, 0.5f);
    std::vector<float> C(M * I, 0.0f);

    int warmup = W, samples = SAMP;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::gemm_ffn(A.data(), gate.data(), up.data(), C.data(), M, I, H);
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::gemm_ffn(A.data(), gate.data(), up.data(), C.data(), M, I, H);
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double flops = 2.0 * M * H * I * 2; // gate + up
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;
    return r;
}

// 9. cross_entropy
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

// 13. gqa_backward
static BenchResult bench_gqa_bwd() {
    BenchResult r{"gqa_backward"};
    set_gpu(true);
    metal_bridge::reconcile_buffers();

    std::vector<float> Q(B * nH * S * HD, 0.1f);
    std::vector<float> K(B * nKV * S * HD, 0.1f);
    std::vector<float> V(B * nKV * S * HD, 0.1f);
    std::vector<float> grad_out(B * S * nH * HD, 0.5f);
    std::vector<float> grad_Q(B * nH * S * HD, 0.0f);
    std::vector<float> grad_K(B * nKV * S * HD, 0.0f);
    std::vector<float> grad_V(B * nKV * S * HD, 0.0f);

    metal_bridge::GQABackwardParams bp;
    bp.batch = B; bp.n_q_heads = nH; bp.n_kv_heads = nKV;
    bp.seq_len = S; bp.head_dim = HD;

    int warmup = W, samples = SAMP_HEAVY;
    for (int i = 0; i < warmup; i++) {
        metal_bridge::gqa_backward(bp, Q.data(), K.data(), V.data(),
                                    grad_out.data(),
                                    grad_Q.data(), grad_K.data(), grad_V.data());
    }
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        metal_bridge::gqa_backward(bp, Q.data(), K.data(), V.data(),
                                    grad_out.data(),
                                    grad_Q.data(), grad_K.data(), grad_V.data());
    }
    r.wall_ms = (now_ms() - t0) / samples;
    double flops = 2.0 * B * nH * S * S * HD * 2;
    r.gflops = flops / (r.wall_ms / 1000.0) / 1e9;
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
    auto fill = [](Tensor &t, float v) { for (auto &x : t.data()) x = v; };
    fill(layer.w_gate, 0.02f); fill(layer.w_up, 0.02f); fill(layer.w_down, 0.02f);
    fill(layer.attn.Wq(), 0.02f); fill(layer.attn.Wk(), 0.02f);
    fill(layer.attn.Wv(), 0.02f); fill(layer.attn.Wo(), 0.02f);

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
    auto fill = [](Tensor &t, float v) { for (auto &x : t.data()) x = v; };
    fill(layer.w_gate, 0.02f); fill(layer.w_up, 0.02f); fill(layer.w_down, 0.02f);
    fill(layer.attn.Wq(), 0.02f); fill(layer.attn.Wk(), 0.02f);
    fill(layer.attn.Wv(), 0.02f); fill(layer.attn.Wo(), 0.02f);

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
        metal_bridge::gemm_proj(data.data(), data.data(), result.data(), M, H, H);
    }
    metal_bridge::reconcile_buffers();
    double t0 = now_ms();
    for (int i = 0; i < samples; i++) {
        // This forces a stepCache miss → upload + copy-back + alloc
        metal_bridge::gemm_proj(data.data(), data.data(), result.data(), M, H, H);
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

    // ── GEMMs ──
    results.push_back(bench_gemm_proj(H,   "QKV (H=1024)"));      // 1024×1024
    results.push_back(bench_gemm_proj(I,   "gate/up (I=2752)"));  // 1024×2752
    results.push_back(bench_gemm_ffn());
    results.push_back(bench_gqa());

    // ── Backward GEMMs ──
    results.push_back(bench_gemm_bwd(H, H, M, "Wq/Wo grad (H,M)"));   // 1024×1024×32768
    results.push_back(bench_gemm_bwd(H, I, M, "Wgate/Wup grad (I,M)"));// 1024×2752×32768
    results.push_back(bench_gemm_bwd(I, H, M, "Wdown grad (I,M)"));    // 2752×1024×32768
    results.push_back(bench_gemm_proj_trans_b(M, H, H, "grad_input (M,H)")); // 32768×1024×1024
    results.push_back(bench_gemm_proj_trans_b(M, H, I, "grad_ffn_in (M,I)"));// 32768×1024×2752
    results.push_back(bench_gqa_bwd());

    // ── Layer-level ──
    results.push_back(bench_layer_fwd());
    results.push_back(bench_layer_bwd());

    // ── Overhead ──
    results.push_back(bench_stepcache_overhead());

    // ── Print table ──
    // Target time per block: what this block needs to achieve for the 24-layer
    // step to complete in 32.4s.  Derived as: 32.4s = 32400ms total for 24 layers.
    // budget_ms = 32400 / 24 = 1350ms per-layer budget.
    // Each block's target is proportional to its current fraction of the total.
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
                  << std::fixed << std::setprecision(3) << std::setw(14) << r.wall_ms
                  << std::fixed << std::setprecision(3) << std::setw(14) << target
                  << std::fixed << std::setprecision(1) << std::setw(14) << r.gflops
                  << std::fixed << std::setprecision(1) << std::setw(14) << r.bw_gbs
                  << "\n";
    }

    // ── Projected step time ──
    std::cout << "\n══════════════════════════════════════════════════════════════\n";
    std::cout << "  PROJECTED ONE-STEP TIME (24 layers)\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";
    // Estimate: per layer cost × 24 + overhead
    double layer_fwd = 0, layer_bwd = 0;
    for (auto &r : results) {
        if (r.label == "one_layer_forward (full)") layer_fwd = r.wall_ms;
        if (r.label == "one_layer_backward (full)") layer_bwd = r.wall_ms;
    }
    double opt_ms = 0;
    for (auto &r : results) {
        if (r.label.find("adamw") != std::string::npos) opt_ms = r.wall_ms;
    }
    // Count: 1 forward + 1 backward per layer + optimizer for all params
    // Parameters: embedding V×H + output_proj H×V + n_layers×(7 weight matrices)
    size_t n_params = V * H + H * V + 24 * (H * I * 3 + H * H + H * nKV * HD * 2 + nH * HD * H);
    double opt_total = opt_ms * n_params / (H * I); // approximate

    double total_ms = 24 * (layer_fwd + layer_bwd) + opt_total;
    std::cout << "  Forward per layer:  " << layer_fwd << " ms\n";
    std::cout << "  Backward per layer: " << layer_bwd << " ms\n";
    std::cout << "  24-layer fwd:       " << 24 * layer_fwd << " ms\n";
    std::cout << "  24-layer bwd:       " << 24 * layer_bwd << " ms\n";
    std::cout << "  Optimizer (approx): " << opt_total << " ms\n";
    std::cout << "  Total per step:     " << total_ms / 1000.0 << " s\n";
    std::cout << "  Target:             < 32.4 s\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    return 0;
}
