Apple Silicon & Metal Compute Architect Skills

Skill 1: Metal Syntax Pre-flight Check

Command Name: check_metal_syntax

Trigger: Whenever modifying .metal, .mm, or .cpp GPU files.

Instruction: Before modifying the C++ bridge, you MUST run ./scripts/check_metal.sh <filename> independently. If the Metal compiler throws an error, fix the syntax before proceeding.

Skill 2: Run and Parse Benchmark Ledger

Command Name: run_benchmark

Trigger: After any successful compilation.

Instruction: Execute scripts/run_bench.sh. Read the BOTTLENECK LEDGER output and identify the kernel with the highest Total (ms). Your next optimization must strictly target that specific bottleneck.

Skill 3: LLDB Crash & Segmentation Fault Triage

Command Name: debug_crash

Trigger: Whenever a segmentation fault, memory corruption, or GPU driver crash occurs during execution.

Instruction: Run the debugging sequence immediately to capture the stack trace and isolate the root cause:

cd ~/dev/large-language-model/build
lldb ./bench_blocks
(lldb) run
# When the crash/sigsegv occurs:
(lldb) bt


Analyze the backtrace frame-by-frame. If it crashes inside Metal command buffer commitments or raw pointer offsets, verify pointer alignments (posix_memalign) and buffer sizing before making changes.

Skill 4: The 5 Laws of Apple Silicon Hardware

Context/Rules:

NO GLOBAL ATOMICS ON LARGE MATRICES: Apple Unified Memory crashes (Lock Convoy) if thousands of threads use atomic_fetch_add_explicit on the same global tensor simultaneously without a stagger effect.

THE REDUNDANT MATH PARADOX: Recomputing lightweight math (like the Softmax denominator / Phase 1) is faster than splitting kernels, because the extra math staggers the memory bus access and prevents lock-step atomic collisions.

COALESCED READS ARE MANDATORY: Never read transposed data directly from global memory. Always load row-wise into threadgroup shared memory, call threadgroup_barrier, and read transposed from SRAM.

TENSOR CORES (simdgroup_matrix): Always use simdgroup_matrix<float, 8, 8> and simdgroup_multiply_accumulate for matrix multiplications. Never use scalar for-loops for GEMMs.

CPU/GPU SYNC: C++ timers will report 0.00ms or mismatch if you do not explicitly wait for the GPU command buffer to complete before stopping the timer.