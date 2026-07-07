// ==============================================================================
// TECHNICAL SPECIFICATION & ARCHITECTURAL REFERENCE: FFN GEMM KERNEL
// ==============================================================================
//
// 1. WHAT IS GEMM?
//    GEMM stands for General Matrix Multiplication. In this kernel, we compute
//    C = A * B, where Matrix A represents input activations [M x K], Matrix B 
//    represents Feed-Forward Weights [K x N] (typically 4H wide), and Matrix C is 
//    the output [M x N] (typically 4H wide). All matrices are in row-major order.
//
// 2. WHO CALLS THIS KERNEL AND WHY?
//    This kernel is invoked by the host C++ engine via the Objective-C++ bridge
//    (MetalBridge.mm). We use it instead of Apple's MPS to achieve zero-overhead 
//    execution and tile specifically for the wide Feed-Forward Network (FFN) shape
//    [M, H] x [H, 4H], maximizing the hardware tensor matrix cores.
//
// 3. HOW DOES THIS FIT IN LLM TRAINING?
//    In a Transformer, the FFN block (containing gate, up, and down projection layers)
//    is the most computationally heavy block in the model, consuming roughly two-thirds
//    of the total FLOPs. Speeding up these wide matrix multiplications directly dictates
//    the model's training throughput (tokens per second).
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
//    - The output Matrix C [M x N] is partitioned into large 2D tiles of size 128 x 128.
//    - A single Threadgroup of 128 threads is assigned to compute one 128 x 128 tile.
//    - To avoid slow VRAM reads, we allocate 32 KB of Shared Threadgroup Memory (L1 cache) 
//      divided into shared_A [128 x 32] (16 KB) and shared_B [32 x 128] (16 KB).
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
//    - Step A: The GPU Global Scheduler assigns a Threadgroup (128 threads) to a Core.
//    - Step B: The Core Scheduler splits the 128 threads into 4 SIMDgroups (SIMD 0, 1, 2, 3)
//              of 32 threads each.
//    - Step C (The K-Loop): We iterate along the inner dimension K in chunks of 32:
//      - The 128 threads collaboratively load data from RAM into shared_A and shared_B.
//      - A threadgroup_barrier forces all execution to halt until L1 memory is populated.
//      - The 4 SIMDgroups are arranged in a 2x2 grid to calculate the 128x128 tile:
//        - SIMD 0: Top-Left [64 x 64]
//        - SIMD 1: Top-Right [64 x 64]
//        - SIMD 2: Bottom-Left [64 x 64]
//        - SIMD 3: Bottom-Right [64 x 64]
//      - Each SIMDgroup loops inside its registers, loading 8x8 slices from L1 into register
//        matrices and accumulating matrix math using simdgroup_multiply_accumulate.
//      - A second barrier ensures all math reading from L1 is complete before L1 is overwritten.
//    - Step D: Once all K iterations are complete, the threads store their register matrices
//              directly back into global RAM at C.
// ==============================================================================

#include <metal_stdlib>
using namespace metal;

// Tiling dimensions defined as compile-time constants.
// This allows the GPU compiler to statically allocate register tiles and L1 cache sizes.
#define TILE_M 128
#define TILE_N 128
#define TILE_K 32


kernel void gemm_ffn(
    device const float* A       [[buffer(0)]], // Matrix A [M x K] (row-major)
    device const float* B       [[buffer(1)]], // Matrix B [K x N] (row-major)
    device float*       C       [[buffer(2)]], // Matrix C [M x N] (row-major)
    constant uint3&     dims    [[buffer(3)]], // dims.x = M, dims.y = N, dims.z = K
    uint2 tg_id   [[threadgroup_position_in_grid]], // Index of the 128x128 tile we are calculating
    uint2 lid     [[thread_position_in_threadgroup]], // Thread coordinates inside the threadgroup (0..31, 0..3)
    uint  simd_id [[simdgroup_index_in_threadgroup]], // SIMD group index (0, 1, 2, or 3)
    uint  lane_id [[thread_index_in_simdgroup]]) // Thread index inside the SIMD group (0..31)
{
  uint M = dims.x;
  uint N = dims.y;
  uint K = dims.z;

  // 1. Allocate Shared Threadgroup Memory (our whiteboard)
  // We allocate static local arrays in the core's high-speed L1 cache using our compile-time tiles:
  // - shared_A: TILE_M x TILE_K elements (128 x 32 floats = 16 KB)
  // - shared_B: TILE_K x TILE_N elements (32 x 128 floats = 16 KB)
  // - Total shared cache occupied = 32 KB per threadgroup
  threadgroup float shared_A[TILE_M][TILE_K];
  threadgroup float shared_B[TILE_K][TILE_N];

  // 2. Define Sub-tile Mapping for each SIMD group
  // We arrange the 4 SIMD groups (0, 1, 2, 3) as a 2x2 grid to compute the 128x128 tile:
  // - SIMD 0: Top-Left [64 x 64]
  // - SIMD 1: Top-Right [64 x 64]
  // - SIMD 2: Bottom-Left [64 x 64]
  // - SIMD 3: Bottom-Right [64 x 64]
  uint sg_row = simd_id / 2; // 0 or 1
  uint sg_col = simd_id % 2; // 0 or 1

  // 3. Initialize Registers to Accumulate the Output
  // Each SIMD group is responsible for a 64x64 sub-tile of C.
  // We divide the 64x64 sub-tile into an 8x8 grid of 8x8 matrix blocks.
  // Each block is held in registers using the simdgroup_matrix helper.
  simdgroup_matrix<float, 8, 8> accum[8][8];
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  // Find the global row/column offset for this threadgroup tile
  uint tile_row_start = tg_id.y * 128;
  uint tile_col_start = tg_id.x * 128;

  // Flatten the thread ID within the threadgroup (from 0 to 127) for collaborative loading
  uint thread_linear_id = simd_id * 32 + lane_id;

  // 4. Slide our window along the K dimension
  for (uint k_offset = 0; k_offset < K; k_offset += 32) {
    
    // COLLABORATIVE LOAD: Load 128x32 slice of A into shared_A
    // Since we have 128 threads and 128x32 = 4096 elements to load:
    // Each thread loads exactly 32 elements. To keep memory reads coalesced,
    // threads load elements in contiguous strides.
    for (uint i = 0; i < 32; ++i) {
      // Linear index of the element this thread is loading
      uint load_idx = thread_linear_id + i * 128;
      uint load_row = load_idx / 32;
      uint load_col = load_idx % 32;

      uint global_row = tile_row_start + load_row;
      uint global_col = k_offset + load_col;

      if (global_row < M && global_col < K) {
        shared_A[load_row][load_col] = A[global_row * K + global_col];
      } else {
        shared_A[load_row][load_col] = 0.0f; // Padding out of bounds
      }
    }

    // COLLABORATIVE LOAD: Load 32x128 slice of B into shared_B
    // We have 128 threads and 32x128 = 4096 elements.
    for (uint i = 0; i < 32; ++i) {
      uint load_idx = thread_linear_id + i * 128;
      uint load_row = load_idx / 128;
      uint load_col = load_idx % 128;

      uint global_row = k_offset + load_row;
      uint global_col = tile_col_start + load_col;

      if (global_row < K && global_col < N) {
        shared_B[load_row][load_col] = B[global_row * N + global_col];
      } else {
        shared_B[load_row][load_col] = 0.0f; // Padding out of bounds
      }
    }

    // Synchronize to make sure all threads have finished loading A and B slices
    // before any thread starts computing with them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Compute Inner Dot Products using hardware 8x8 matrix units
    // Loop over the K-slice width (32 elements). We move in steps of 8.
    for (uint k_step = 0; k_step < 32; k_step += 8) {
      
      // Load 8x8 slices of A and B from shared memory into registers
      simdgroup_matrix<float, 8, 8> mat_A[8];
      simdgroup_matrix<float, 8, 8> mat_B[8];

      // Load 8 rows of A for this SIMD group's rows
      for (int r = 0; r < 8; ++r) {
        // Load an 8x8 matrix from shared_A
        // Arguments: destination register, source pointer, stride, offset
        simdgroup_load(mat_A[r], &shared_A[sg_row * 64 + r * 8][k_step], 32, ulong2(0));
      }

      // Load 8 columns of B for this SIMD group's columns
      for (int c = 0; c < 8; ++c) {
        // Load an 8x8 matrix from shared_B
        simdgroup_load(mat_B[c], &shared_B[k_step][sg_col * 64 + c * 8], 128, ulong2(0));
      }

      // Perform hardware matrix multiply-accumulate (Tensor core equivalents)
      // Accumulates: accum[r][c] += mat_A[r] * mat_B[c]
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
          simdgroup_multiply_accumulate(accum[r][c], mat_A[r], mat_B[c], accum[r][c]);
        }
      }
    }

    // Synchronize again to make sure all threads have finished reading from shared memory
    // before we overwrite it with the next K-slice in the next iteration.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // 6. Write final outputs back to Unified RAM (Matrix C)
  // Each SIMD group stores its 64x64 sub-tile of accumulators.
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      uint global_row = tile_row_start + sg_row * 64 + r * 8;
      uint global_col = tile_col_start + sg_col * 64 + c * 8;

      // Make sure we are within the boundaries of Matrix C
      if (global_row < M && global_col < N) {
        // Store the 8x8 matrix result directly into global RAM at C
        simdgroup_store(accum[r][c], &C[global_row * N + global_col], N, ulong2(0));
      }
    }
  }
}
