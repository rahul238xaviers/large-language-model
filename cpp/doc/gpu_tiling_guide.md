# Apple Silicon GPU Tiling and Parameter Selection Guide

This guide helps developers understand how to select optimal parameters (tile sizes, thread counts, and grid sizes) when launching custom GPU GEMM (General Matrix Multiply) kernels on Apple Silicon. 
---

## 1. Hardware Profile: Apple M3 Ultra (80 Core GPU 512 GB Unified Memory)

To optimize code for the GPU, we must align our execution model with the physical layout of the silicon:

*   **Total GPU Cores**: 80 physical cores.
*   **ALUs (Execution Lanes) Per Core**: 128 ALUs.
    *   This means each core executes exactly **128 threads physically in parallel** at any given clock cycle.
*   **SIMD Units Per Core**: 4 physical SIMD execution slots.
    *   Apple Silicon uses a native SIMD width of **32 threads** (called a SIMDgroup).
    *   Therefore, the 128 execution lanes per core are divided into:
        `128 ALUs / 32 threads per SIMD unit = 4 physical SIMD execution units`.
    *   **Important**: A core can have **more than 4 SIMDgroups loaded** at a time (e.g., 8 or more from multiple tiles). However, only **4 SIMDgroups can physically execute in any single clock cycle** because there are only 128 ALUs. The remaining SIMDgroups sit in a ready queue — they use this waiting time to prefetch data from memory and prepare their registers. When one of the 4 active SIMDgroups stalls (e.g., waiting for a memory read), the core instantly swaps in a ready SIMDgroup from the queue. This is how the GPU hides memory latency.
*   **L1 Cache (Shared memory) Per Core**: 64 KB.
    *   This high-speed cache is shared among all active Threadgroups loaded onto that core.
*   **Max Registers Per Thread**: 256 registers.
    *   Each thread has a private register file. Exceeding 256 causes "register spilling" to slow VRAM.

---

## 2. What the Programmer Chooses

When writing a GPU GEMM kernel, the programmer must decide **3 tile dimensions**:

1. **TILE_M**: Height of one tile (how many rows of Matrix C one threadgroup computes).
2. **TILE_N**: Width of one tile (how many columns of Matrix C one threadgroup computes).
3. **TILE_K**: Inner dimension step size (how many elements of K we load per loop iteration).

These three values, combined with the matrix dimensions (M, K, N), determine everything else.

### Why 3 Tiles, Not 2?

The output Matrix C is 2D (M x N), so we partition it into a 2D grid of tiles using TILE_M and TILE_N.

But to compute each tile, we must multiply slices of A and B along the inner dimension K. Loading the entire K dimension at once would be too large for the L1 cache. So we split K into steps of size TILE_K and loop through them.

Example: If K = 4096 and TILE_K = 32, the loop runs `4096 / 32 = 128` iterations. In each iteration, we load:
*   A slice of A: `TILE_M x TILE_K` elements
*   A slice of B: `TILE_K x TILE_N` elements

These two slices are what we store in the L1 cache (shared memory). Matrix C is **never** stored in L1 — each thread accumulates its portion of C in its private registers.

---

## 3. Derived Attributes (Calculated from Programmer Choices + Hardware)

All 5 attributes below are **calculated** — the programmer does not set them directly.

| # | Attribute | Formula | What It Tells Us |
| :--- | :--- | :--- | :--- |
| 1 | **L1 Cache Needed** | `(TILE_M * TILE_K + TILE_K * TILE_N) * 4 bytes` | How much shared memory one tile needs for its A and B slices |
| 2 | **Core Occupancy** | `64 KB / L1 Cache Needed` (rounded down) | How many tiles fit on one core simultaneously |
| 3 | **Register Reuse** | `(TILE_M * TILE_N) / (TILE_M + TILE_N)` | How many math operations per L1 cache read |
| 4 | **Number of Tiles** | `(M / TILE_M) * (N / TILE_N)` | Total threadgroups launched across the GPU |
| 5 | **Threads per Tile** | Typically `TILE_M` for square tiles | Workers assigned to each tile |

### Attribute Validation Ranges

| Attribute | Minimum | Optimal | Maximum | What Happens if Out of Range |
| :--- | :--- | :--- | :--- | :--- |
| **L1 Cache Needed** | - | 16-32 KB | 64 KB (hard limit) | Tile won't fit on the core |
| **Core Occupancy** | 2 | 4 | - | Below 2: no latency hiding |
| **Register Reuse** | 16 | 32 | 128 | Below 16: ALUs stall waiting for data |
| **Private Registers** | - | 64-128 | 256 (hard limit) | Above 256: register spilling to VRAM |

---

## 4. Tile Size Profiles (with TILE_K = 32, i.e. the inner dimension)

| Metric | 128 x 128 Tile | 64 x 64 Tile | 32 x 32 Tile |
| :--- | :--- | :--- | :--- |
| **L1 Cache Needed** | 32 KB | 16 KB | 8 KB |
| **Core Occupancy** | 2 tiles per core | 4 tiles per core | 8 tiles per core |
| **Register Reuse** | 64 | 32 | 16 |
| **Threads per Tile** | 128 (4 SIMDgroups) | 64 (2 SIMDgroups) | 32 (1 SIMDgroup) |
| **Registers per Thread** | 128 | 64 | 32 |

---

## 5. Worked Example: M = 2048, K = 4096, N = 2048

Let's trace how the three candidate sizes perform on the M3 Ultra for this specific matrix shape:
*   Matrix A: `2048 x 4096`
*   Matrix B: `4096 x 2048`
*   Output Matrix C: `2048 x 2048` (This is the matrix we partition into tiles)

### Candidate A: 128 x 128 Tiles
1.  **Grid Calculations (Partitioning Matrix C)**:
    *   Vertical Tile Count (along height M of C): `2048 (M) / 128 (TILE_M) = 16 tiles`
    *   Horizontal Tile Count (along width N of C): `2048 (N) / 128 (TILE_N) = 16 tiles`
    *   Total Tiles: `16 * 16 = 256`
2.  **Core Occupancy**:
    *   Each tile consumes `32 KB` of L1 cache.
    *   Only `2 tiles` fit on a core at one time.
3.  **Waves to Execute on 80 Cores**:
    *   The GPU can hold `80 cores * 2 tiles = 160 tiles` concurrently in flight.
    *   Total waves required: `ceil(256 / 160) = 2 waves`.
    *   *Wave 1*: 160 tiles running (80 cores at 100% occupancy).
    *   *Wave 2*: 96 tiles running (48 cores active, 32 cores sitting idle).
4.  **Register Reuse**:
    *   Very high (64 calculations per L1 cache read).

### Candidate B: 64 x 64 Tiles
1.  **Grid Calculations (Partitioning Matrix C)**:
    *   Vertical Tile Count (along height M of C): `2048 (M) / 64 (TILE_M) = 32 tiles`
    *   Horizontal Tile Count (along width N of C): `2048 (N) / 64 (TILE_N) = 32 tiles`
    *   Total Tiles: `32 * 32 = 1024`
2.  **Core Occupancy**:
    *   Each tile consumes `16 KB` of L1 cache.
    *   `4 tiles` fit on a core at one time.
3.  **Waves to Execute on 80 Cores**:
    *   The GPU can hold `80 cores * 4 tiles = 320 tiles` concurrently in flight.
    *   Total waves required: `ceil(1024 / 320) = 4 waves`.
    *   *Wave 1, 2, 3*: 320 tiles running (80 cores at 100% occupancy).
    *   *Wave 4*: 64 tiles running (16 cores active at full occupancy, or spread evenly for load-balancing).
    *   Because we have 4 waves of 4 tiles per core, the scheduler has an abundance of ready tiles to swap in to mask VRAM latency.
4.  **Register Reuse**:
    *   High (32 calculations per L1 cache read).

### Candidate C: 32 x 32 Tiles
1.  **Grid Calculations**:
    *   Vertical Tile Count (Height): `2048 / 32 = 64 tiles`
    *   Horizontal Tile Count (Width): `2048 / 32 = 64 tiles`
    *   Total Tiles: `64 * 64 = 4096`
2.  **Core Occupancy**:
    *   Each tile consumes `8 KB` of L1 cache.
    *   `8 tiles` fit on a core.
3.  **Waves to Execute on 80 Cores**:
    *   The GPU holds `80 cores * 8 tiles = 640 tiles` concurrently.
    *   Total waves required: `ceil(4096 / 640) = 7 waves`.
4.  **Register Reuse**:
    *   Poor (only 16 calculations per L1 cache read). The ALUs will stall waiting for data to load from L1 into registers.

---

## 6. Optimization Verdict

For this matrix dimension (`2048 x 4096 x 2048`):

*   **`128 x 128`** is highly optimized for register reuse, but its second wave leaves 32 cores completely idle, and 2 tiles per core limits memory latency masking.
*   **`32 x 32`** has high occupancy but is bottlenecked by the L1-to-register bandwidth due to poor register reuse.
*   **`64 x 64`** is the **most optimal selection**. It provides a perfect balance of high register reuse (32) and high occupancy (4 tiles per core), keeping all 80 cores fully busy across 4 well-balanced execution waves.
