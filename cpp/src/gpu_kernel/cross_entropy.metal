// ==============================================================================
// KERNEL: cross_entropy
// WHAT: Performs fused Softmax + Cross Entropy Loss + Gradient generation on GPU.
// WHY:  Replaces the CPU multithreaded (std::thread) Loss.cpp which previously
//       iterated over 3.2 billion elements on CPU every step.
// SHAPE:
//   logits     : float[total_tokens * vocab_size]
//   targets    : uint32[total_tokens]
//   loss_out   : float[1] (atomic accumulation)
//   grad_logits: float[total_tokens * vocab_size]
// THREAD LAYOUT:
//   2D grid of threadgroups: (256, total_tokens, 1)
//   Each threadgroup handles 1 token position (b,s) using 256 parallel threads.
// ==============================================================================

#include <metal_stdlib>
using namespace metal;

// Helper function for threadgroup float reduction (max)
inline float threadgroup_max(float val, threadgroup float* shared_mem, uint tid) {
    shared_mem[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared_mem[0];
}

// Helper function for threadgroup float reduction (sum)
inline float threadgroup_sum(float val, threadgroup float* shared_mem, uint tid) {
    shared_mem[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared_mem[0];
}

kernel void cross_entropy(
    device const bfloat* logits     [[buffer(0)]],
    device const uint*  targets     [[buffer(1)]],
    device atomic_float* loss_out   [[buffer(2)]],
    device bfloat*      grad_logits [[buffer(3)]],
    constant uint&      vocab_size  [[buffer(4)]],
    constant uint&      total_tokens[[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint t   [[threadgroup_position_in_grid]]
) {
    if (t >= total_tokens) return;

    threadgroup float shared_mem[256];
    device const bfloat* token_logits = logits + t * vocab_size;
    device bfloat* token_grads = grad_logits + t * vocab_size;
    uint target_id = targets[t];
    // Guard: an out-of-range target would index past the row (GPU fault).
    if (target_id >= vocab_size) return;

    // --- Pass 1: Local Max ---
    float local_max = -INFINITY;
    for (uint v = tid; v < vocab_size; v += 256) {
        local_max = max(local_max, token_logits[v]);
    }
    float group_max = threadgroup_max(local_max, shared_mem, tid);

    // --- Pass 2: Sum Exp ---
    float local_sum_exp = 0.0f;
    for (uint v = tid; v < vocab_size; v += 256) {
        local_sum_exp += exp(token_logits[v] - group_max);
    }
    float group_sum_exp = threadgroup_sum(local_sum_exp, shared_mem, tid);

    // --- Pass 3: Loss calculation & Grad generation ---
    if (tid == 0) {
        float target_logit = token_logits[target_id];
        float target_prob = exp(target_logit - group_max) / group_sum_exp;
        float loss_val = -log(max(target_prob, 1e-15f)) / float(total_tokens);
        atomic_fetch_add_explicit(loss_out, loss_val, memory_order_relaxed);
    }

    float inv_total = 1.0f / float(total_tokens);
    for (uint v = tid; v < vocab_size; v += 256) {
        float prob = exp(token_logits[v] - group_max) / group_sum_exp;
        float is_target = (v == target_id) ? 1.0f : 0.0f;
        token_grads[v] = (bfloat)((prob - is_target) * inv_total);
    }
}
