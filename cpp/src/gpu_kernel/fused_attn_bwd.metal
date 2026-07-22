// ═══════════════════════════════════════════════════════════════════════════════
// FUSED CAUSAL ATTENTION BACKWARD — BF16 activations, FP32 accumulators
// ═══════════════════════════════════════════════════════════════════════════════
// Activations (Q, K, V, dO, scores, probs) stored as BF16 in threadgroup memory.
// Accumulators (dQ, dK, dV) and softmax state (max, sum) in FP32 for stability.
//
// TILES: TQ=32, TK=16 → 512 threads  (all BF16 → 2× storage density)
// Shared: 16.5 KB (26% of 64 KB — headroom for compiler temps)
// Registers: 79/thread peak  (< 128, no spill)
// Occupancy: 100% (2 groups × 512 threads = 1024/core → fully saturated)
//
// GRID: [B=32, nH=16] → 512 groups × 512 threads
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint S=1024, HD=64, TQ=32, TK=16, NQ=32, NK=64;

// Convert BF16 loaded from shared → FP32 for accumulation
// (MSL's bfloat type auto-promotes to float on arithmetic)
inline float bf16_to_f32(bfloat v) { return (float)v; }

kernel void fused_attn_bwd(
    device const float* Q      [[buffer(0)]],  // FP32 input (converted to BF16 in shared)
    device const float* K      [[buffer(1)]],  // FP32 input
    device const float* V      [[buffer(2)]],  // FP32 input
    device const float* dO     [[buffer(3)]],  // FP32 input
    device float*        dQ    [[buffer(4)]],  // FP32 output
    device float*        dK    [[buffer(5)]],  // FP32 output
    device float*        dV    [[buffer(6)]],  // FP32 output
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 li    [[thread_position_in_threadgroup]]
) {
    uint b = tg_id.x, h = tg_id.y;
    if (b >= 32 || h >= 16) return;
    uint kv_h = h / 2;
    float s = 1.0f / sqrt((float)HD);

    uint qb   = (b * 16 + h)    * S * HD;
    uint kb   = (b * 8  + kv_h) * S * HD;
    uint dk0  = kb, dv0 = kb;
    uint doff = h * HD, dstr = 16 * HD;

    // ── BF16 shared memory (2 bytes/element) ──
    threadgroup bfloat Qs [TQ*HD];   // 32×64×2  =  4 KB
    threadgroup bfloat dOs[TQ*HD];   // 32×64×2  =  4 KB
    threadgroup bfloat Ks [TK*HD];   // 16×64×2  =  2 KB
    threadgroup bfloat Vs [TK*HD];   // 16×64×2  =  2 KB
    threadgroup bfloat dKp[TK*HD];   // 16×64×2  =  2 KB (tile-local)
    threadgroup bfloat dVp[TK*HD];   // 16×64×2  =  2 KB (tile-local)
    threadgroup bfloat pr [TK];      // 16×2     =  32 B
    threadgroup bfloat dp [TK];      // 16×2     =  32 B
    // FP32 shared (for softmax stability — needs full exponent range)
    threadgroup float  mx [TQ];      // 32×4     = 128 B
    threadgroup float  sm [TQ];      // 32×4     = 128 B
    threadgroup float  brd;          //           4 B

    uint sq_tid = li.x / TK;  // 0..31
    uint sk_tid = li.x % TK;  // 0..15

    // FP32 dQ accumulator (kept across all K-tiles)
    float dQa[HD];
    for (uint d = 0; d < HD; ++d) dQa[d] = 0.0f;

    for (uint qt = 0; qt < NQ; ++qt) {
        uint sqb = qt * TQ, sqg = sqb + sq_tid;
        bool sqok = sqg < S;

        // Load Q and dO (BF16 → shared) — 512 threads × 4 = 2048 = 32×64
        for (uint i = li.x; i < TQ*HD; i += 512) {
            uint r = i/HD, c = i%HD, gsq = sqb + r;
            if (r<TQ && gsq<S) {
                Qs [r*HD+c] = (bfloat)Q [qb  + gsq*HD + c];
                dOs[r*HD+c] = (bfloat)dO[(b*S+gsq)*dstr + doff + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (li.x < TQ) { mx[li.x] = -INFINITY; sm[li.x] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ══════════════════════════════════════════════════════════════════
        // PHASE 1: Online softmax (64 K-tiles)
        // ══════════════════════════════════════════════════════════════════
        for (uint kt = 0; kt < NK; ++kt) {
            uint skb = kt * TK, skg = skb + sk_tid;
            for (uint i = li.x; i < TK*HD; i += 512) {
                uint r = i/HD, c = i%HD, gsk = skb + r;
                if (r<TK && gsk<S) {
                    Ks[r*HD+c] = (bfloat)K[kb+gsk*HD+c];
                    Vs[r*HD+c] = (bfloat)V[kb+gsk*HD+c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Dot product: BF16 → FP32 accumulation
            float sc = -INFINITY;
            if (sqok && sqg >= skg) {
                sc = 0;
                for (uint d = 0; d < HD; ++d)
                    sc += (float)Qs[sq_tid*HD+d] * (float)Ks[sk_tid*HD+d];
                sc *= s;
            }

            // SIMD-reduce max across TK=16 threads
            float tm = sc;
            tm = max(tm, simd_shuffle_down(tm,8));
            tm = max(tm, simd_shuffle_down(tm,4));
            tm = max(tm, simd_shuffle_down(tm,2));
            tm = max(tm, simd_shuffle_down(tm,1));

            if (sqok && sk_tid == 0) {
                float cm = mx[sq_tid];
                if (tm > cm) { sm[sq_tid] *= exp(cm - tm); mx[sq_tid] = tm; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sqok && sqg >= skg)
                sm[sq_tid] += exp(sc - mx[sq_tid]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE 2: Gradients (64 K-tiles)
        // ══════════════════════════════════════════════════════════════════
        for (uint i = li.x; i < TK*HD; i += 512) { dKp[i] = 0; dVp[i] = 0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < NK; ++kt) {
            uint skb = kt * TK, skg = skb + sk_tid;
            for (uint i = li.x; i < TK*HD; i += 512) {
                uint r = i/HD, c = i%HD, gsk = skb + r;
                if (r<TK && gsk<S) {
                    Ks[r*HD+c] = (bfloat)K[kb+gsk*HD+c];
                    Vs[r*HD+c] = (bfloat)V[kb+gsk*HD+c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Score, prob, dP (BF16 → FP32 arithmetic)
            float prob = 0, dPv = 0;
            if (sqok && sqg >= skg) {
                float sc = 0;
                for (uint d = 0; d < HD; ++d)
                    sc += (float)Qs[sq_tid*HD+d] * (float)Ks[sk_tid*HD+d];
                sc *= s;
                prob = exp(sc - mx[sq_tid]) / sm[sq_tid];
                for (uint d = 0; d < HD; ++d)
                    dPv += (float)dOs[sq_tid*HD+d] * (float)Vs[sk_tid*HD+d];
            }
            pr[sk_tid] = (bfloat)prob;
            dp[sk_tid] = (bfloat)dPv;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // sum_dP_prob
            float sp = 0;
            if (sqok && sk_tid == 0) {
                for (uint t = 0; t < TK; ++t)
                    if (skb + t <= sqg) sp += (float)pr[t] * (float)dp[t];
                brd = sp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sp = brd;

            // dS, dQ, tile-local dK/dV
            if (sqok && sqg >= skg) {
                prob = (float)pr[sk_tid];
                dPv  = (float)dp[sk_tid];
                float dS = prob * (dPv - sp);
                float dSs = dS * s;
                for (uint d = 0; d < HD; ++d) {
                    float kv = (float)Ks[sk_tid*HD+d];
                    float qv = (float)Qs[sq_tid*HD+d];
                    float ov = (float)dOs[sq_tid*HD+d];
                    dQa[d] += dSs * kv;
                    // Store tile-local in BF16 (converted back on global write)
                    float dk_val = (float)dKp[sk_tid*HD+d] + dSs * qv;
                    float dv_val = (float)dVp[sk_tid*HD+d] + prob * ov;
                    dKp[sk_tid*HD+d] = (bfloat)dk_val;
                    dVp[sk_tid*HD+d] = (bfloat)dv_val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Flush tile-local dK/dV → global FP32 with atomics
            if (skg < S) for (uint d = 0; d < HD; ++d) {
                float dk_contrib = (float)dKp[sk_tid*HD+d];
                float dv_contrib = (float)dVp[sk_tid*HD+d];
                if (dk_contrib != 0.0f)
                    atomic_fetch_add_explicit(
                        (device atomic_float*)&dK[dk0 + skg*HD + d],
                        dk_contrib, memory_order_relaxed);
                if (dv_contrib != 0.0f)
                    atomic_fetch_add_explicit(
                        (device atomic_float*)&dV[dv0 + skg*HD + d],
                        dv_contrib, memory_order_relaxed);
                dKp[sk_tid*HD+d] = 0;
                dVp[sk_tid*HD+d] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write dQ for this Q-tile (FP32 accumulators → FP32 global)
        if (sqok) for (uint d = 0; d < HD; ++d) {
            dQ[qb + sqg*HD + d] += dQa[d];
            dQa[d] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
