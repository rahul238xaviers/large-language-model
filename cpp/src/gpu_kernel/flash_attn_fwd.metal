// ═══════════════════════════════════════════════════════════════════════════════
// flash_attn_fwd — FlashAttention-2 forward: fused QK^T→softmax→PV
// ═══════════════════════════════════════════════════════════════════════════════
// No N×N global score/probability matrix materialization.
// Online softmax via running max + denominator trick.
//
// Grid:        [B, nH, 1]  — one threadgroup per (batch, query_head)
// Threads/TG:  256  (8 warps × 32 lanes)
//
// Block tiling:
//   Br = 32  (Q rows per tile)   Bc = 32  (K/V rows per tile)
//   Each warp handles Br/8 = 4 Q-rows; each lane handles 1 K-column.
//
// Online softmax state kept in register per warp-row:
//   m[i] — running max,  l[i] — running denominator sum
//   O[i] — FP32 output accumulator (4×64 elements per warp × 4 rows)
//
// Shared memory: 20.25 KB  (63% of 32 KB)
//   Qs[32×64] BF16  4096 B
//   Ks[32×64] BF16  4096 B
//   Vs[32×64] BF16  4096 B
//   Oa[32×64] FP32  8192 B  (written once at end of Q-tile)
//
// Causal mask: scores[k > q] = -INF  (handled by lane index vs lane's assigned row)
// GQA: kv_h = q_head / (nH/nKV)
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint S=1024, HD=64, Br=32, Bc=32;
constant uint NQ=S/Br, NK=S/Bc;     // 32 tiles each
constant uint WL=32;

kernel void flash_attn_fwd(
    device const float* Q      [[buffer(0)]],  // [B, nH,  S, HD]
    device const float* K      [[buffer(1)]],  // [B, nKV, S, HD]
    device const float* V      [[buffer(2)]],  // [B, nKV, S, HD]
    device float*       O      [[buffer(3)]],  // [B, nH,  S, HD]
    constant uint*      n_heads_ptr [[buffer(4)]],
    constant uint*      n_kv_ptr    [[buffer(5)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 li_p  [[thread_position_in_threadgroup]]
) {
    uint b = tg_id.x, h = tg_id.y;
    uint n_heads = *n_heads_ptr, n_kv = *n_kv_ptr;
    if (b >= 32 || h >= n_heads) return;
    uint kv_h = h / (n_heads / n_kv);
    float scale = 1.0f / sqrt((float)HD);

    uint li = li_p.x, sg_id = li / WL, ln_id = li % WL;
    uint QR = Br / 8;  // 4 Q-rows per warp

    threadgroup bfloat Qs[2048];   // 32×64 BF16
    threadgroup bfloat Ks[2048];   // 32×64 BF16
    threadgroup bfloat Vs[2048];   // 32×64 BF16
    threadgroup float  Oa[2048];   // 32×64 FP32

    uint q_off  = (b * n_heads + h)   * S * HD;
    uint kv_off = (b * n_kv + kv_h)   * S * HD;

    float reg_m[4];   // QR = 4
    float reg_l[4];

    // ── Outer loop: iterate over Q blocks ──
    for (uint qt = 0; qt < NQ; ++qt) {
        uint q_base = qt * Br;
        uint gqr0 = q_base + sg_id * QR;  // global row of this warp's first row

        // ── Load Q-tile: float4 coalesced → shared BF16 ──
        // 256 threads: each loads Br*HD/256 = 8 elements = 2 float4's
        for (uint i = li * 2; i < Br * HD; i += 256 * 2) {
            uint r = i / HD, c = i % HD;
            uint gr = q_base + r;
            if (gr < S) { Qs[r*HD+c] = (bfloat)Q[q_off + gr*HD + c]; }
            if (i+1 < Br*HD) {
                uint r2 = (i+1)/HD, c2 = (i+1)%HD;
                uint gr2 = q_base + r2;
                if (gr2 < S) { Qs[r2*HD+c2] = (bfloat)Q[q_off + gr2*HD + c2]; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Init registers
        for (uint qr = 0; qr < QR; ++qr) {
            for (uint d = ln_id; d < HD; d += WL) Oa[(sg_id*QR+qr)*HD + d] = 0.0f;
            reg_m[qr] = -INFINITY;
            reg_l[qr] = 0.0f;
        }

        // ── Inner loop: K/V tiles (only up to qt for causal) ──
        for (uint kt = 0; kt <= qt && kt < NK; ++kt) {
            uint k_base = kt * Bc;

            // Load K,V tiles: float4 coalesced → shared BF16
            for (uint i = li * 2; i < Bc * HD; i += 256 * 2) {
                uint r = i / HD, c = i % HD;
                uint gr = k_base + r;
                if (gr < S) {
                    Ks[r*HD + c] = (bfloat)K[kv_off + gr*HD + c];
                    Vs[r*HD + c] = (bfloat)V[kv_off + gr*HD + c];
                }
                if (i+1 < Bc*HD) {
                    uint r2 = (i+1)/HD, c2 = (i+1)%HD;
                    uint gr2 = k_base + r2;
                    if (gr2 < S) {
                        Ks[r2*HD + c2] = (bfloat)K[kv_off + gr2*HD + c2];
                        Vs[r2*HD + c2] = (bfloat)V[kv_off + gr2*HD + c2];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── Process each Q-row in this warp ──
            for (uint qr = 0; qr < QR; ++qr) {
                uint gqr = gqr0 + qr;
                if (gqr >= S) continue;

                // 1. S[j] = Q[gqr] · K[j] for j = 0..Bc-1 (lane = column)
                uint kj = k_base + ln_id;
                float s_val = -INFINITY;
                if (kj <= gqr && kj < S) {
                    s_val = 0.0f;
                    for (uint d = 0; d < HD; ++d)
                        s_val += (float)Qs[(sg_id*QR+qr)*HD + d] * (float)Ks[ln_id*HD + d];
                    s_val *= scale;
                }

                // 2. Warp rowmax
                float wmax = s_val;
                for (uint mask = 16; mask > 0; mask >>= 1)
                    wmax = max(wmax, simd_shuffle_down(wmax, mask));
                wmax = simd_shuffle(wmax, 0);

                // 3. Online softmax
                float m_new = max(reg_m[qr], wmax);
                float scale_old = exp(reg_m[qr] - m_new);
                reg_m[qr] = m_new;

                // 4. P[j] = exp(S[j] - m_new)
                float p_val = (kj <= gqr && kj < S) ? exp(s_val - m_new) : 0.0f;
                float p_sum = p_val;
                for (uint mask = 16; mask > 0; mask >>= 1)
                    p_sum += simd_shuffle_down(p_sum, mask);

                // 5. Rescale O and l, accumulate
                reg_l[qr] = reg_l[qr] * scale_old + simd_shuffle(p_sum, 0);

                // 6. O[qr] += P[j] × V[j]
                uint o_row = (sg_id*QR+qr) * HD;
                for (uint d = ln_id; d < HD; d += WL) {
                    Oa[o_row + d] *= scale_old;
                    float vv = (kj <= gqr && kj < S) ? (float)Vs[ln_id*HD + d] : 0.0f;
                    Oa[o_row + d] += p_val * vv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Write normalized O to global ──
        for (uint qr = 0; qr < QR; ++qr) {
            uint gqr = gqr0 + qr;
            if (gqr >= S) continue;
            uint o_row = (sg_id*QR+qr) * HD;
            for (uint d = ln_id; d < HD; d += WL)
                O[q_off + gqr*HD + d] = Oa[o_row + d] / reg_l[qr];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
