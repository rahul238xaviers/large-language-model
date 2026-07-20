// ==============================================================================
// KERNEL: gemm_proj_trans_b
// WHAT: Performs Matrix Multiplication with transposed B: C [M x N] = A [M x K] * B^T [K x N]
// WHY:  In backward propagation, gradient w.r.t input activation is computed as:
//       grad_input [M, K] = grad_output [M, N] * W^T [N, K]
//       Matrix B is stored in memory as [N x K]. This kernel reads B transposed directly on GPU,
//       eliminating CPU weight matrix transposes (vDSP_mtrans) and CPU cblas_sgemm calls.
// ==============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

kernel void gemm_proj_trans_b(
    device const float* A [[buffer(0)]], // Shape [M x K] in memory
    device const float* B [[buffer(1)]], // Shape [N x K] in memory, accessed as B^T [K x N]
    device float*       C [[buffer(2)]], // Shape [M x N] output
    constant uint3&  dims [[buffer(3)]], // x = M, y = N, z = K
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_A[TILE_M][TILE_K];
    threadgroup float shared_B[TILE_K][TILE_N];

    simdgroup_matrix<float, 8, 8> accum[8][4];
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint tile_start_col = TILE_N * tg_id.x;
    uint tile_start_row = TILE_M * tg_id.y;
    uint thread_linear_id = simd_id * 32 + lane_id;

    for (uint k = 0; k < dims.z; k += TILE_K) {
        // Load A [64 x 32]
        for (uint i = 0; i < 32; ++i) {
            uint load_idx = thread_linear_id + i * 64;
            uint load_row = load_idx / 32;
            uint load_col = load_idx % 32;

            uint global_row = tile_start_row + load_row;
            uint global_col = k + load_col;

            if (global_row < dims.x && global_col < dims.z) {
                shared_A[load_row][load_col] = A[global_row * dims.z + global_col];
            } else {
                shared_A[load_row][load_col] = 0.0f;
            }
        }

        // Load B transposed into shared_B [32 x 64]
        // B in memory is [N x K] (dims.y x dims.z). B^T[load_row, load_col] = B[global_col, global_row]
        for (uint i = 0; i < 32; ++i) {
            uint load_idx = thread_linear_id + i * 64;
            uint load_row = load_idx / 64;
            uint load_col = load_idx % 64;

            uint global_row = k + load_row;         // index along K
            uint global_col = tile_start_col + load_col; // index along N

            if (global_row < dims.z && global_col < dims.y) {
                shared_B[load_row][load_col] = B[global_col * dims.z + global_row];
            } else {
                shared_B[load_row][load_col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_step = 0; k_step < TILE_K; k_step += 8) {
            simdgroup_matrix<float, 8, 8> mat_A[8];
            simdgroup_matrix<float, 8, 8> mat_B[4];

            for (int r = 0; r < 8; ++r) {
                simdgroup_load(mat_A[r], &shared_A[r * 8][k_step], TILE_K, ulong2(0));
            }
            for (int c = 0; c < 4; ++c) {
                simdgroup_load(mat_B[c], &shared_B[k_step][simd_id * 32 + c * 8], TILE_N, ulong2(0));
            }
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 4; ++c) {
                    simdgroup_multiply_accumulate(accum[r][c], mat_A[r], mat_B[c], accum[r][c]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write Matrix C [M x N] to global memory
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            uint global_row = tile_start_row + r * 8;
            uint global_col = tile_start_col + simd_id * 32 + c * 8;

            if (global_row < dims.x && global_col < dims.y) {
                simdgroup_store(accum[r][c], &C[global_row * dims.y + global_col], dims.y, ulong2(0));
            }
        }
    }
}
