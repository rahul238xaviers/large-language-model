// ==============================================================================
// Grouped Query Attention (GQA) Fused Tiled Forward Kernel
// ==============================================================================

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

#include <metal_stdlib>
using namespace metal;

struct GQAParams {
  uint batch;
  uint n_q_heads;
  uint n_kv_heads;
  uint seq_len;
  uint head_dim;
};

kernel void gemm_gqa(
    device const bfloat* Q      [[buffer(0)]],
    device const bfloat* K      [[buffer(1)]],
    device const bfloat* V      [[buffer(2)]],
    device bfloat*       out    [[buffer(3)]],
    device float*        L_out  [[buffer(4)]],
    constant GQAParams&  gqa    [[buffer(5)]],
    uint3                tg_id  [[threadgroup_position_in_grid]],
    uint3                ti     [[thread_position_in_threadgroup]]
) {
    uint li = ti.x;
    // Threadgroup mapping:
    // - tg_id.x -> Sequence block index (each block computes 32 query tokens)
    // - tg_id.y -> Batch index
    // - tg_id.z -> Query head index
    uint block_y = tg_id.x;
    uint b = tg_id.y;
    uint h = tg_id.z;

    uint s_q = block_y * 32 + li;

    uint group_size = gqa.n_q_heads / gqa.n_kv_heads;
    uint h_kv = h / group_size;

    float scale = 1.0f / sqrt((float)gqa.head_dim);

    // Q pointer for this thread's token
    device const bfloat* q_ptr = Q + (b * gqa.n_q_heads * gqa.seq_len + h * gqa.seq_len + s_q) * gqa.head_dim;

    // Cache Q in thread registers (enforce zero-spill constraint for HD=64)
    float thread_q[64];
    if (s_q < gqa.seq_len) {
        for (uint d = 0; d < 64; d += 4) {
            bfloat4 q_val = *((device const bfloat4*)(q_ptr + d));
            thread_q[d + 0] = (float)q_val[0];
            thread_q[d + 1] = (float)q_val[1];
            thread_q[d + 2] = (float)q_val[2];
            thread_q[d + 3] = (float)q_val[3];
        }
    }

    // Allocate threadgroup shared memory for K and V tiles
    // Tile size = 32 sequence tokens * 64 head_dim
    threadgroup float shared_K[32 * 64];
    threadgroup float shared_V[32 * 64];

    // Base pointers for K and V heads
    uint kv_common_factor = (b * gqa.n_kv_heads * gqa.seq_len + h_kv * gqa.seq_len) * gqa.head_dim;
    device const bfloat* k_base = K + kv_common_factor;
    device const bfloat* v_base = V + kv_common_factor;

    // Initialize online softmax
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float accum_val[64];
    for (uint d = 0; d < 64; d++) {
        accum_val[d] = 0.0f;
    }

    // The maximum active sequence token processed in this threadgroup
    uint seq_len_limit = min((block_y + 1) * 32 - 1, gqa.seq_len - 1);

    // Loop through keys/values in blocks of 32
    for (uint tile_k = 0; tile_k <= seq_len_limit / 32; tile_k++) {
        uint s_k_base = tile_k * 32;
        uint global_s_k = s_k_base + li;

        // Cooperatively load a tile of K and V into shared memory using vectorized float4 reads
        if (global_s_k < gqa.seq_len) {
            device const bfloat* k_ptr = k_base + global_s_k * gqa.head_dim;
            device const bfloat* v_ptr = v_base + global_s_k * gqa.head_dim;
            for (uint d = 0; d < 64; d += 4) {
                bfloat4 k_val = *((device const bfloat4*)(k_ptr + d));
                bfloat4 v_val = *((device const bfloat4*)(v_ptr + d));
                shared_K[li * 64 + d + 0] = (float)k_val[0];
                shared_K[li * 64 + d + 1] = (float)k_val[1];
                shared_K[li * 64 + d + 2] = (float)k_val[2];
                shared_K[li * 64 + d + 3] = (float)k_val[3];
                shared_V[li * 64 + d + 0] = (float)v_val[0];
                shared_V[li * 64 + d + 1] = (float)v_val[1];
                shared_V[li * 64 + d + 2] = (float)v_val[2];
                shared_V[li * 64 + d + 3] = (float)v_val[3];
            }
        } else {
            for (uint d = 0; d < 64; d++) {
                shared_K[li * 64 + d] = 0.0f;
                shared_V[li * 64 + d] = 0.0f;
            }
        }

        // Synchronize threads so all loads into shared memory are complete
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process keys in this tile
        if (s_q < gqa.seq_len) {
            uint max_k_in_tile = min(uint(32), s_q - s_k_base + 1);
            for (uint k_idx = 0; k_idx < max_k_in_tile; k_idx++) {
                // Compute dot product Q * K
                float sum = 0.0f;
                for (uint d = 0; d < 64; d++) {
                    sum += thread_q[d] * shared_K[k_idx * 64 + d];
                }
                float score = sum * scale;

                // Online softmax update
                float m_old = max_val;
                if (score > max_val) {
                    max_val = score;
                    float exp_scale = exp(m_old - max_val);
                    sum_exp = sum_exp * exp_scale + 1.0f;
                    for (uint d = 0; d < 64; d++) {
                        accum_val[d] = accum_val[d] * exp_scale + shared_V[k_idx * 64 + d];
                    }
                } else {
                    float exp_term = exp(score - max_val);
                    sum_exp += exp_term;
                    for (uint d = 0; d < 64; d++) {
                        accum_val[d] += exp_term * shared_V[k_idx * 64 + d];
                    }
                }
            }
        }

        // Synchronize again before the next tile overwrites shared memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write the normalized result to VRAM
    if (s_q < gqa.seq_len) {
        device bfloat* out_ptr = out + (b * gqa.seq_len * gqa.n_q_heads + s_q * gqa.n_q_heads + h) * gqa.head_dim;
        for (uint d = 0; d < 64; d += 4) {
            bfloat4 out_val;
            out_val[0] = (bfloat)(accum_val[d + 0] / sum_exp);
            out_val[1] = (bfloat)(accum_val[d + 1] / sum_exp);
            out_val[2] = (bfloat)(accum_val[d + 2] / sum_exp);
            out_val[3] = (bfloat)(accum_val[d + 3] / sum_exp);
            *((device bfloat4*)(out_ptr + d)) = out_val;
        }
        L_out[b * gqa.n_q_heads * gqa.seq_len + h * gqa.seq_len + s_q] = max_val + log(sum_exp);
    }
}
