#include<metal_stdlib>
using namespace metal;

// CMake may define TILE_M/TILE_N/TILE_K via -D flags for other kernels;
// undefine them here so our local const declarations take precedence.
#ifdef TILE_M
#undef TILE_M
#endif
#ifdef TILE_N
#undef TILE_N
#endif
#ifdef TILE_K
#undef TILE_K
#endif

kernel void gemm_proj(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float*       C       [[buffer(2)]],
    constant uint3&     dims    [[buffer(3)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 lid     [[thread_position_in_threadgroup]],
    uint  simd_id [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {
    const uint TILE_M = 64;
    const uint TILE_N = 64;
    const uint TILE_K = 32;

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

        for (uint i = 0; i < 32; ++i) {
            uint load_idx = thread_linear_id + i * 64;
            uint load_row = load_idx / 64;
            uint load_col = load_idx % 64;
            uint global_row = k + load_row;
            uint global_col = tile_start_col + load_col;
            if (global_row < dims.z && global_col < dims.y) {
                shared_B[load_row][load_col] = B[global_row * dims.y + global_col];
            } else {
                shared_B[load_row][load_col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_step = 0; k_step < 32; k_step += 8) {
            simdgroup_matrix<float, 8, 8> mat_A[8];
            simdgroup_matrix<float, 8, 8> mat_B[4];

            for (int r = 0; r < 8; ++r) {
                simdgroup_load(mat_A[r], &shared_A[r * 8][k_step], 32, ulong2(0));
            }
            for (int c = 0; c < 4; ++c) {
                simdgroup_load(mat_B[c], &shared_B[k_step][simd_id * 32 + c * 8], 64, ulong2(0));
            }
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 4; ++c) {
                    simdgroup_multiply_accumulate(accum[r][c], mat_A[r], mat_B[c], accum[r][c]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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
