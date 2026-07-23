// ═══════════════════════════════════════════════════════════════════════════════
// compute_loss — minimal loss kernel (no gradient write)
// ═══════════════════════════════════════════════════════════════════════════════
// Reads logits and targets, computes cross-entropy loss, atomically adds to
// loss_out.  No gradient buffer needed — avoids the 6.1 GB gradient write that
// can stall the batch command buffer.
//
// Grid: [total_tokens, 1, 1], 256 threads per TG
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

inline float tg_max(float val, threadgroup float* mem, uint tid) {
    mem[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) mem[tid] = max(mem[tid], mem[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return mem[0];
}

inline float tg_sum(float val, threadgroup float* mem, uint tid) {
    mem[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) mem[tid] += mem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return mem[0];
}

kernel void compute_loss(
    device const bfloat* logits     [[buffer(0)]],
    device const uint*   targets    [[buffer(1)]],
    device atomic_float* loss_out   [[buffer(2)]],
    constant uint&       vocab_size [[buffer(3)]],
    constant uint&       total_tokens [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint t   [[threadgroup_position_in_grid]]
) {
    if (t >= total_tokens) return;

    threadgroup float shared_mem[256];
    device const bfloat* tk_logits = logits + t * vocab_size;
    uint target_id = targets[t];

    float local_max = -INFINITY;
    for (uint v = tid; v < vocab_size; v += 256)
        local_max = max(local_max, (float)tk_logits[v]);
    float gmax = tg_max(local_max, shared_mem, tid);

    float local_sum = 0.0f;
    for (uint v = tid; v < vocab_size; v += 256)
        local_sum += exp((float)tk_logits[v] - gmax);
    float gsum = tg_sum(local_sum, shared_mem, tid);

    if (tid == 0) {
        float target_logit = (float)tk_logits[target_id];
        float prob = exp(target_logit - gmax) / gsum;
        float loss_val = -log(max(prob, 1e-15f)) / float(total_tokens);
        atomic_fetch_add_explicit(loss_out, loss_val, memory_order_relaxed);
    }
}
