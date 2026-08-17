#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>

// Forward declarations for Objective-C protocol types
#ifdef __OBJC__
@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;
#else
typedef struct objc_object *id;
#endif

namespace metal_bridge {

// ── Global Metal state (extern — same instance across all translation units) ──
/// Returns true if a batch command buffer is active (inside begin/end_scope).
inline bool is_batch_active() { return false; }  // replaced by batchCommandBuffer

void initialize();
bool is_available();

// ── Async scope management ──────────────────────────────────────────
// Replaces per-kernel commit+wait with batched encoding:
//   begin_scope();      // creates a command buffer, sets batchActive
//   gemm_bf16(...);     // encodes into active buffer (no commit)
//   gemm_bf16(...);     // more encodes
//   end_scope();        // commits once, waits, runs copy-back
void begin_scope();
void end_scope();
// Backward compat aliases (test files)
inline void start_batch() { begin_scope(); }
inline void commit_batch() { end_scope(); }

// ── Universal BF16 GEMM ─────────────────────────────────────────────
// C[M,N] = A[M,K] @ B[K,N]  (or with transpose flags)
// A, B, C are BF16.  FP32 internal accumulation via simdgroup_matrix.
// Replaces cblas_sgemm for QKV, MLP, and Output Projection.
void gemm_bf16(const void *A, const void *B, void *C,
               size_t M, size_t N, size_t K,
               bool transA = false, bool transB = false,
               bool is_forward = false);

// ── Small-batch down-projection GEMM ─────────────────────────────────
// C[M,N] = A[M,K] @ B[K,N]  (B stored as N×K, loaded with coalesced access)
// Specialized for M=32, K=1024, N=2752 (down-projection backward shape).
void gemm_proj_down(const void *A, const void *B, void *C,
                    size_t M, size_t N, size_t K);

// ── Coalesced weight-gradient GEMM ──────────────────────────────────
// C[M,N] = A_transposed[M,K] × B[K,N]  (A stored as K×M, B as K×N)
// Specialized for K=32 with trA=true coalesced load.
void gemm_bwd_weight(const void *A, const void *B, void *C,
                     size_t M, size_t N, size_t K);

void fused_swiglu_gemm(const void *gate_proj, const void *up_proj,
                       const void *B, void *C,
                       size_t M, size_t N, size_t K);

// ── FlashAttention forward (fused QK^T→softmax→PV, no N×N intermediates) ──
void flash_attn_fwd(const void *Q, const void *K, const void *V, void *O,
                    size_t batch, size_t n_heads, size_t n_kv,
                    size_t seq_len, size_t head_dim);

// ── Fused attention backward (single kernel, no materialized intermediates) ──
void fused_attn_bwd(const void *Q, const void *K, const void *V,
                    const void *dO, void *dQ, void *dK, void *dV,
                    size_t batch, size_t n_heads, size_t n_kv,
                    size_t seq_len, size_t head_dim);

// ── GPU element-wise kernels (BF16 in/out via pointer cast) ─────────
void rms_norm_forward(const float *input, float *output, const float *weight,
                      float eps, size_t num_rows, size_t dims);
void rms_norm_backward(const float *grad_output, const float *input,
                       const float *weight, float *grad_input,
                       float *grad_weight, float eps, size_t num_rows, size_t dims);
void swiglu_backward(const float *grad_output, const float *gate,
                     const float *up, float *grad_gate, float *grad_up, size_t n);
void rope_backward(float *grad, const float *cos_table, const float *sin_table,
                   size_t batch, size_t heads, size_t seq_len, size_t head_dim);
// ── GPU wrapper registration (called from PagedBuffer on alloc/free) ──
void register_gpu_wrapper(const float* ptr, void** wrapper_loc);
void unregister_gpu_wrapper(const float* ptr);

// ── GPU-native zero-fill and copy (MTLBlitCommandEncoder — no CPU memset) ──
void fill_zero_async(float *data, size_t bytes);
void copy_buffer_async(float *dst, size_t dst_bytes, const float *src, size_t src_bytes);

void residual_add(float *a, const float *b, size_t n);
void fused_add_norm(float *x_residual, const float *residual,
                    const float *weight, float *output,
                    size_t num_rows, size_t D, float eps);
void fused_backward_add_norm(const void *grad_output, const void *input,
                              const void *weight, const void *residual,
                              void *grad_input,
                              size_t num_rows, size_t D, float eps);
void embedding_forward(const uint32_t *token_ids, const float *embedding_table,
                       float *output, size_t total_tokens, size_t hidden_dim,
                       size_t vocab_size);
void cross_entropy(const void *logits, const uint32_t *targets,
                   float *loss_out, float *grad_logits,
                   size_t total_tokens, size_t vocab_size);
struct AdamWStepParams {
  float lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2;
  uint32_t n;
};

void adamw_step(float *param, const float *grad, float *m, float *v,
                const AdamWStepParams &params);

// ── Legacy / deprecated ─────────────────────────────────────────────
// These exist for backward compat but are replaced by gemm_bf16 + async:
struct GQAParams { uint32_t batch, n_q_heads, n_kv_heads, seq_len, head_dim; };

void gemm_gqa(const GQAParams &params, const float *q, const float *k,
              const float *v, float *out_gqa, float *L_out = nullptr);
void gemm_backward(const float *a_transposed, const float *b, float *c,
                   size_t M, size_t N, size_t K);
void gemm_proj_trans_b(const float *a, const float *b_transposed, float *c,
                       size_t M, size_t N, size_t K);
void reshape_to_4d(const float *src, float *dst,
                   size_t batch, size_t n_heads, size_t seq_len, size_t head_dim);
void reshape_to_3d(const float *src, float *dst,
                   size_t batch, size_t n_heads, size_t seq_len, size_t head_dim);
void rope_forward(float *q, float *k,
                  const float *cos_table, const float *sin_table,
                  size_t batch, size_t q_heads, size_t kv_heads,
                  size_t seq_len, size_t head_dim);
void clear_step_cache();
void reconcile_buffers();
float get_last_loss();
void execute_in_autoreleasepool(std::function<void()> func);
void start_step_trace();
void stop_step_trace();

extern double accum_gpu_time_ms;
extern double last_scope_gpu_time_ms;
extern double accum_cpu_time_ms;
extern size_t count_gpu_calls;
extern size_t count_cpu_calls;
void reset_profile_stats();

} // namespace metal_bridge
