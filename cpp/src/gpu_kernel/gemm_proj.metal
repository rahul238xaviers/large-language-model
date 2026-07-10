// ==============================================================================
// TECHNICAL SPECIFICATION & ARCHITECTURAL REFERENCE: PROJECTION GEMM KERNEL
// ==============================================================================
//
// 1. WHAT IS GEMM?
//    GEMM stands for General Matrix Multiplication. In this kernel, we compute
//    C = A * B, where Matrix A represents input activations [M x K], Matrix B 
//    represents projection weights [K x N], and Matrix C is the output [M x N].
//    All matrices are stored in row-major contiguous memory.
//
// 2. WHO CALLS THIS KERNEL AND WHY?
//    This kernel is invoked by the host C++ engine via the Objective-C++ bridge
//    (MetalBridge.mm). We use it instead of Apple's MPS (Metal Performance Shaders)
//    to achieve zero-overhead execution and fine-grained control over cache sizes,
//    allowing us to tile specifically for the projection shape [M, H] x [H, H].
//
// 3. HOW DOES THIS FIT IN LLM TRAINING?
//    In a Transformer architecture, the projection layers (such as QKV projections 
//    and output projection layers) constitute a massive portion of the training
//    workload. This kernel accelerates both the forward inference/training projections
//    and the backward gradient updates by keeping intermediate data local to the core.
//
// 4. WHAT IS MSL AND HOW IS IT TRIGGERED?
//    MSL (Metal Shading Language) is a unified GPU programming language based on C++14.
//    At build time, MSL source files are compiled by CMake (via xcrun) into `.air` 
//    intermediate bytecode and bundled into a `default.metallib` binary archive.
//    At runtime, the host CPU loads this library, compiles it into binary machine 
//    instructions (MTLComputePipelineState), sets the GPU pointer parameters, 
//    and dispatches a grid of threads to trigger execution.
//
// 5. GPU HARDWARE STRUCTURE (Apple Silicon M3 Ultra):
//    - The GPU features 80 physical GPU Cores.
//    - Each GPU Core contains 4 physical SIMD units (ALUs).
//    - Each SIMD unit executes a "SIMDgroup" of exactly 32 threads in physical lockstep.
//    - At any given clock cycle, each core executes 4 SIMDgroups (128 threads) 
//      physically in parallel, while managing a queue of up to 32 SIMDgroups (1024 threads)
//      to hide memory latency.
//
// 6. MATRIX LAYOUT & WORKLOAD PARTITIONING:
//    - The output Matrix C [M x N] is partitioned into 2D tiles of size 64 x 64.
//    - A single Threadgroup of 64 threads is assigned to compute one 64 x 64 tile.
//    - To avoid slow VRAM reads, we allocate 16 KB of Shared Threadgroup Memory (L1 cache) 
//      divided into shared_A [64 x 32] and shared_B [32 x 64].
//    - Threads load contiguous blocks of 32 floats (128 bytes) in parallel, matching the
//      GPU's memory coalescing bus width.
//
// 7. ARITHMETIC PERFORMANCE (Per Clock Cycle):
//    - Inside each SIMDgroup, the threads execute Fused Multiply-Add (FMA) instructions
//      (d = a * b + c) directly on the physical hardware matrix cores (AMX/Tensor cores).
//    - Each FMA instruction performs 2 floating-point operations (1 multiply, 1 add) 
//      in a single clock cycle, maximizing hardware occupancy.
//
// 8. THE LIFECYCLE OF THE WORKLOAD (Thread & SIMDgroup Workflow):
//    - Step A: The GPU Global Scheduler assigns a Threadgroup (64 threads) to a Core.
//    - Step B: The Core Scheduler splits the 64 threads into SIMDgroup 0 (threads 0-31)
//              and SIMDgroup 1 (threads 32-63).
//    - Step C (The K-Loop): We iterate along the inner dimension K in chunks of 32:
//      - The 64 threads collaboratively load data from RAM into shared_A and shared_B.
//      - A threadgroup_barrier forces all execution to halt until L1 memory is populated.
//      - SIMDgroup 0 loads data from L1 and computes the left half of the tile (64x32).
//      - SIMDgroup 1 loads data from L1 and computes the right half of the tile (64x32).
//      - Both SIMDgroups run matrix math inside their registers.
//      - A second barrier ensures all math reading from L1 is complete before L1 is overwritten.
//    - Step D: Once all K iterations are complete, the threads store their register matrices
//              directly back into global RAM at C.
// ==============================================================================

#include<metal_stdlib>
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

#define SIMD_MAX_THREADS 32

kernel void gemm_proj(
    device const float* A       [[buffer(0)]], // Matrix A [M x K] (row-major)
    device const float* B       [[buffer(1)]], // Matrix B [K x N] (row-major)
    device float*       C       [[buffer(2)]], // Matrix C [M x N] (row-major)
    constant uint3&     dims    [[buffer(3)]], // dims.x = M, dims.y = N, dims.z = K
    uint2 tg_id   [[threadgroup_position_in_grid]], // Index of the 64x64 tile we are calculating
    uint2 lid     [[thread_position_in_threadgroup]], // Thread coordinates inside the threadgroup
    uint  simd_id [[simdgroup_index_in_threadgroup]], // SIMD group index (0, 1)
    uint  lane_id [[thread_index_in_simdgroup]] // Thread index inside the SIMD group (0..31)
) {
    threadgroup float shared_A[TILE_M][TILE_K];
    threadgroup float shared_B[TILE_K][TILE_N];

    // Initialize simdgroup matrix accumulators (8x8 blocks)
    // SIMD 0 computes columns 0..31 (4 blocks), SIMD 1 computes columns 32..63 (4 blocks)
    simdgroup_matrix<float, 8, 8> accum[8][4];
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint tile_start_col = TILE_N * tg_id.x;
    uint tile_start_row = TILE_M * tg_id.y;

    uint thread_linear_id = simd_id * 32 + lane_id;

    // Slide our window along the K dimension in chunks of TILE_K (32)
    for (uint k = 0; k < dims.z; k += TILE_K) {
        
        // COLLABORATIVE LOAD: Load 64x32 slice of A into shared_A
        // We have 64 threads, 64x32 = 2048 elements. Each thread loads 32 elements.
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

        // COLLABORATIVE LOAD: Load 32x64 slice of B into shared_B
        // We have 64 threads, 32x64 = 2048 elements. Each thread loads 32 elements.
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

        // First Barrier: wait for loading to finish
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute using 8x8 matrix hardware units
        for (uint k_step = 0; k_step < 32; k_step += 8) {
            simdgroup_matrix<float, 8, 8> mat_A[8];
            simdgroup_matrix<float, 8, 8> mat_B[4];

            // Load 8 rows of A for this SIMD group (handles all 64 rows of A)
            for (int r = 0; r < 8; ++r) {
                simdgroup_load(mat_A[r], &shared_A[r * 8][k_step], 32, ulong2(0));
            }

            // Load 4 columns of B for this SIMD group's columns (SIMD 0: 0..31, SIMD 1: 32..63)
            for (int c = 0; c < 4; ++c) {
                simdgroup_load(mat_B[c], &shared_B[k_step][simd_id * 32 + c * 8], 64, ulong2(0));
            }

            // Multiply and accumulate
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 4; ++c) {
                    simdgroup_multiply_accumulate(accum[r][c], mat_A[r], mat_B[c], accum[r][c]);
                }
            }
        }

        // Second Barrier: wait for compute to finish before loading next K-slice
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write accumulators back to Matrix C in global RAM/Unified Memory
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

    
