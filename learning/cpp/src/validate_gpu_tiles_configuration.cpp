#include <iostream>
#include <cmath>
#include <vector>

int main() {

  // =========================================================================
  // SECTION 1: Programmer Choices (Change these to test different configs)
  // =========================================================================

  // The matrices to multiply: C = A * B
  // Matrix A: rows = M, cols = K
  // Matrix B: rows = K, cols = N
  // Matrix C (output): rows = M, cols = N
  std::vector<int> MATRIX_A = {1024, 4096};  // {rows, cols} = {M, K}
  std::vector<int> MATRIX_B = {4096, 1024};  // {rows, cols} = {K, N}

  // Tile sizes chosen by the programmer
  int TILE_M = 64;   // Height of one tile (rows of C per tile)
  int TILE_N = 64;   // Width of one tile (cols of C per tile)
  int TILE_K = 32;   // Inner dimension step size (loop step along K)

  // =========================================================================
  // SECTION 2: Hardware Constants (Fixed for M3 Ultra 80-Core GPU)
  // =========================================================================

  const size_t THREADS_PER_SIMD = 32;  // A SIMDgroup is always 32 threads (hardwired)
  const size_t L1_CACHE_PER_CORE = 64; // 64 KB of shared memory per core
  const size_t GPU_CORES = 80;         // Total physical GPU cores
  const size_t MAX_REGISTERS_PER_THREAD = 256; // Max registers available per thread

  // =========================================================================
  // SECTION 3: Derived Values (Calculated from Sections 1 and 2)
  // =========================================================================

  // Output Matrix C dimensions (derived from A and B)
  int M = MATRIX_A[0];  // Rows of A = Rows of C
  int N = MATRIX_B[1];  // Cols of B = Cols of C
  int K = MATRIX_A[1];  // Cols of A = Rows of B (inner dimension)
  std::vector<int> MATRIX_C = {M, N};

  std::cout << "=== Matrix Dimensions ===" << std::endl;
  std::cout << "Matrix A: " << M << " x " << K << std::endl;
  std::cout << "Matrix B: " << K << " x " << N << std::endl;
  std::cout << "Matrix C (output): " << M << " x " << N << std::endl;
  std::cout << std::endl;

  // --- Attribute 1: L1 Cache Needed (per tile) ---
  // WHAT: How much shared memory one tile needs to load its slices of A and B
  // WHY: We load a slice of A (TILE_M x TILE_K) and a slice of B (TILE_K x TILE_N)
  //      into the L1 cache. Matrix C is NOT stored in L1 — it lives in registers.
  size_t tile_budget_bytes = ((size_t)TILE_M * TILE_K + (size_t)TILE_K * TILE_N) * 4;
  size_t tile_budget_kb = tile_budget_bytes / 1024;

  std::cout << "=== Attribute 1: L1 Cache Needed ===" << std::endl;
  std::cout << "Shared A slice: " << TILE_M << " x " << TILE_K
            << " = " << TILE_M * TILE_K << " floats" << std::endl;
  std::cout << "Shared B slice: " << TILE_K << " x " << TILE_N
            << " = " << TILE_K * TILE_N << " floats" << std::endl;
  std::cout << "L1 Cache needed per tile: " << tile_budget_kb << " KB" << std::endl;

  if (tile_budget_kb > L1_CACHE_PER_CORE) {
    std::cout << "[FAIL] Tile exceeds the " << L1_CACHE_PER_CORE
              << " KB L1 cache limit per core!" << std::endl;
  } else {
    std::cout << "[OK] Tile fits within L1 cache." << std::endl;
  }
  std::cout << std::endl;

  // --- Attribute 2: Core Occupancy ---
  // WHAT: How many tiles (threadgroups) can fit on one core at the same time
  // WHY: More tiles per core = better latency hiding (when one tile stalls
  //      waiting for memory, the core can switch to another tile)
  size_t core_occupancy = (tile_budget_kb > 0) ? L1_CACHE_PER_CORE / tile_budget_kb : 0;

  std::cout << "=== Attribute 2: Core Occupancy ===" << std::endl;
  std::cout << "Tiles fitting per core: " << core_occupancy << std::endl;

  if (core_occupancy < 2) {
    std::cout << "[WARN] Only " << core_occupancy
              << " tile(s) per core — no latency hiding possible." << std::endl;
  } else {
    std::cout << "[OK] " << core_occupancy
              << " tiles per core — good for latency hiding." << std::endl;
  }
  std::cout << std::endl;

  // --- Attribute 3: Register Reuse Ratio ---
  // WHAT: How many multiply-accumulate operations we do per value loaded from L1
  // WHY: Higher ratio = ALUs stay busy. Lower ratio = ALUs stall waiting for data.
  size_t register_reuse_ratio = ((size_t)TILE_M * TILE_N) / ((size_t)TILE_M + TILE_N);

  std::cout << "=== Attribute 3: Register Reuse Ratio ===" << std::endl;
  std::cout << "Register reuse ratio: " << register_reuse_ratio << std::endl;

  if (register_reuse_ratio < 16) {
    std::cout << "[WARN] Ratio too low — ALUs will stall waiting for L1 data."
              << std::endl;
  } else if (register_reuse_ratio > 128) {
    std::cout << "[WARN] Ratio unrealistically high — tile may be too large."
              << std::endl;
  } else {
    std::cout << "[OK] Good register reuse." << std::endl;
  }
  std::cout << std::endl;

  // --- Attribute 4: Number of Tiles (Threadgroups) ---
  // WHAT: Total tiles needed to cover the entire output Matrix C
  // WHY: The number of rows in C (M) tells us the vertical tile count,
  //      the number of columns in C (N) tells us the horizontal tile count.
  size_t vertical_tiles = M / TILE_M;
  size_t horizontal_tiles = N / TILE_N;
  size_t total_tiles = vertical_tiles * horizontal_tiles;

  std::cout << "=== Attribute 4: Number of Tiles ===" << std::endl;
  std::cout << "Vertical tiles (M / TILE_M): " << vertical_tiles << std::endl;
  std::cout << "Horizontal tiles (N / TILE_N): " << horizontal_tiles << std::endl;
  std::cout << "Total tiles (threadgroups): " << total_tiles << std::endl;

  if (M % TILE_M != 0) {
    std::cout << "[WARN] M (" << M << ") is not divisible by TILE_M ("
              << TILE_M << ")!" << std::endl;
  }
  if (N % TILE_N != 0) {
    std::cout << "[WARN] N (" << N << ") is not divisible by TILE_N ("
              << TILE_N << ")!" << std::endl;
  }
  if (K % TILE_K != 0) {
    std::cout << "[WARN] K (" << K << ") is not divisible by TILE_K ("
              << TILE_K << ")!" << std::endl;
  }
  std::cout << std::endl;

  // --- Attribute 5: Threads in Each Tile and Private Registers ---
  // WHAT: Each tile has a certain number of output elements (TILE_M * TILE_N).
  //       We need enough threads to compute them, and each thread stores its
  //       share of the output in private registers.
  // WHY: The threads in a tile must be a multiple of 32 (SIMDgroup size).
  //      The registers per thread must not exceed the hardware limit (256).
  size_t tile_elements = (size_t)TILE_M * TILE_N;

  // We calculate threads per tile from the tile size:
  // Each SIMDgroup of 32 threads computes a sub-block of the tile.
  // For square tiles: threads_per_tile typically equals TILE_M or TILE_N.
  // The key constraint: registers_per_thread must be <= MAX_REGISTERS_PER_THREAD.
  //
  // We try threads_per_tile = TILE_M (common choice for square tiles),
  // then verify the register count is within budget.
  size_t threads_per_tile = TILE_M;
  size_t simd_groups_per_tile = threads_per_tile / THREADS_PER_SIMD;
  size_t registers_per_thread = tile_elements / threads_per_tile;

  std::cout << "=== Attribute 5: Threads and Registers ===" << std::endl;
  std::cout << "Tile elements (TILE_M * TILE_N): " << tile_elements << std::endl;
  std::cout << "Threads in each tile: " << threads_per_tile << std::endl;
  std::cout << "SIMDgroups in each tile: " << simd_groups_per_tile << std::endl;
  std::cout << "Private registers per thread: " << registers_per_thread << std::endl;

  if (threads_per_tile % THREADS_PER_SIMD != 0) {
    std::cout << "[FAIL] Threads per tile (" << threads_per_tile
              << ") must be a multiple of " << THREADS_PER_SIMD << "!" << std::endl;
  }
  if (registers_per_thread > MAX_REGISTERS_PER_THREAD) {
    std::cout << "[FAIL] Register spilling! Each thread needs " << registers_per_thread
              << " registers but hardware limit is " << MAX_REGISTERS_PER_THREAD
              << "." << std::endl;
  } else {
    std::cout << "[OK] Registers fit within hardware budget." << std::endl;
  }
  std::cout << std::endl;

  // --- Waves Calculation ---
  // WHAT: How many "waves" the GPU needs to process all tiles
  // WHY: The GPU can only run (GPU_CORES * core_occupancy) tiles at a time.
  //      If we have more tiles than that, it takes multiple waves.
  size_t tiles_in_flight = GPU_CORES * core_occupancy;
  size_t waves = (tiles_in_flight > 0)
                     ? (size_t)std::ceil((double)total_tiles / tiles_in_flight)
                     : 0;

  std::cout << "=== Waves Calculation ===" << std::endl;
  std::cout << "Tiles in flight at once: " << tiles_in_flight << std::endl;
  std::cout << "Total waves needed: " << waves << std::endl;

  size_t last_wave_tiles = total_tiles - (waves - 1) * tiles_in_flight;
  size_t last_wave_active_cores = (last_wave_tiles + core_occupancy - 1) / core_occupancy;

  if (waves > 1) {
    std::cout << "Last wave tiles: " << last_wave_tiles << std::endl;
    std::cout << "Last wave active cores: " << last_wave_active_cores
              << " / " << GPU_CORES << std::endl;

    if (last_wave_active_cores < GPU_CORES / 2) {
      std::cout << "[WARN] Last wave uses less than half the GPU cores — "
                << "consider a smaller tile size for better load balancing."
                << std::endl;
    }
  }
  std::cout << std::endl;

  // --- Total Threads ---
  size_t total_threads = total_tiles * threads_per_tile;
  std::cout << "=== Summary ===" << std::endl;
  std::cout << "Total threads launched: " << total_threads << std::endl;

  return 0;
}