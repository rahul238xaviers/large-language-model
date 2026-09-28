# BUG REPORT: Intermittent heap corruption ("bad weak table") during CPP GPU training

**Handoff document for a fresh agent.** Read this fully before touching anything.
This describes a memory-corruption bug that crashes the training pipeline after
~26-52 steps. Several real bugs have already been found and fixed (section 5) —
do not "re-discover" those; focus on the remaining root cause (section 8).

---

## 1. Summary

The C++/Metal LLM pre-training pipeline (`./build/run_trainer`) crashes after
~26-52 training steps (batch-dependent) with:

```
objc[PID]: bad weak table at 0x<addr>. This may be a runtime bug or a memory error somewhere else.
```

This is an **Objective-C runtime heap-corruption abort**. It is preceded by a
**one-time GPU page fault** at step 0 (forward pass) that appears in *every* run:

```
[GPU] command buffer error: Caused GPU Address Fault Error (0000000b:kIOGPUCommandBufferCallbackErrorPageFault)
```

The step-0 fault is benign in the short term (loss is correct afterward), but the
underlying out-of-bounds access accumulates heap corruption until the ObjC weak
table is hit. The exact crash step is heap-layout-dependent (varies between runs).

## 2. Symptoms (in order)

1. Step 0, forward scope: `[GPU] command buffer error: ... kIOGPUCommandBufferCallbackErrorPageFault` (once per run).
2. Steps continue correctly (loss = `ln(vocab)` = 11.516 at init, checkpoints save/reload).
3. After ~26-52 steps: `objc[...]: bad weak table` → process aborts (exit 134).

Note: the loss being correct at step 0 proves the faulting kernel writes its
needed output *before* faulting, OR faults on a redundant read — the fault is
not (yet) corrupting the forward outputs.

## 3. Reproduction

From repo root:

```bash
# batch 4 (lighter): crashes ~step 26-52 (~5.5 s/step)
./build/run_trainer --batch_size 4 --max_steps 500 --checkpoint_interval 500

# batch 32 (heavy): crashes ~step 19 (~35 s/step)
./build/run_trainer --batch_size 32 --max_steps 500 --checkpoint_interval 500
```

Data dependency: `data/datasets/rust/train.bin` (100 MB = 25M tokens,
24,390 sequences of 1025). Only supports ~762 steps @ batch 32.

## 4. Environment

- macOS, Apple **M3 Ultra** (512 GB unified memory)
- Metal + MetalPerformanceShaders, Objective-C++ (`MetalBridge.mm`)
- Build: `cmake -S . -B build && cmake --build build --target run_trainer`
- Branch: `feature/optimize-fused-attn` (heavy uncommitted WIP)

Not capacity-related: system RAM 98% free; GPU `recommendedMaxWorkingSetSize`
= 498 GB; run uses ~29 GB. Not data-related: all token/target ids are valid
(checked: max id 100255 < vocab 100277).

## 5. ALREADY-FIXED BUGS (do not re-investigate; verify if unsure)

### 5.1 `fused_swiglu_gemm.metal` — unguarded bfloat4 tail loads (PRIMARY corruption source)
File: `cpp/src/gpu_kernel/fused_swiglu_gemm.metal`, `load_tile`.
The guard checked only the tile **start** (`gc < Kdim`), then did a `bfloat4`
(4-element) load that ran past the row end. With `Kdim = 2730` (not a multiple
of 4), the last row's tail load read 2 elements **past the entire buffer** → GPU
page fault → heap corruption.
**Fix applied:** vector path requires `gc + 4 <= Kdim` (A/gate/up) and
`gc + 4 <= N` (B/w_down); otherwise falls to the bounds-checked scalar path.
A step-0 GPU fault *still* occurs after this fix, so this was necessary but not
sufficient — see section 8.

### 5.2 `Transformer.cpp` `accumulate_embedding_grads` — BF16 read as FP32 (correctness + OOB read)
File: `cpp/src/Transformer.cpp` (~line 596).
`grad_h` is a **BF16** tensor in the GPU path; `Tensor::operator()` reads via
`data_float()` (4 bytes/elem), so it over-read the 2-byte buffer by 2× and
returned garbage embedding gradients (the model was not actually learning
embeddings — loss was frozen at 11.5156 before this fix).
**Fix applied:** convert `grad_h` to FP32 before indexed reads.

### 5.3 `MetalBridge.mm` `get_or_create_buffer` — cache + size-pool double-tracking
File: `cpp/src/gpu_kernel/MetalBridge.mm` (~line 744).
Fallback buffers were **both** cached in `persistent_fallback_cache` (keyed by
raw `ptr`) **and** pushed to `activeAllocationsThisStep` (returned to the size
pool at `end_scope`). When a temporary reused the same heap address, the cache
returned a buffer already handed to another tensor → two tensors sharing one
`MTLBuffer` → corruption.
**Fix applied:** `is_persistent` → cached only; otherwise → pool only. Never both.

### 5.4 `MetalBridge.mm` — NoCopy buffers pooled and reused for wrong pointers
File: `cpp/src/gpu_kernel/MetalBridge.mm` (~line 748, `end_scope` ~line 1970).
Page-aligned `newBufferWithBytesNoCopy` wrappers are tied to one host pointer's
memory. They were pooled and later reused for a **different** pointer → GPU
writes went into stale/freed memory.
**Fix applied:** `nocopyAllocationsThisStep` vector; wrappers are
`CFRelease`d at `end_scope` (never pooled). This extended the crash from
~step 26 → ~52 (batch 4), confirming it contributed but was not the only source.

### 5.5 Defensive bounds checks added
- `embedding_forward.metal`: `if (token_id >= vocab_size) { output=0; return; }`
  (new `constant uint& vocab_size` at buffer(5); dispatch updated).
- `cross_entropy.metal`: `if (target_id >= vocab_size) return;`

## 6. Kernel work done earlier (correct, keep)

- `gemm_bwd_weight.metal` was rewritten as a coalesced multi-K-tile kernel
  (BM=64, BN=128, BK=32, double-buffered) and `gemm_backward` now routes all
  weight-gradients through it (K = token dimension). Verified correct (L2 < 1e-2)
  and ~15% faster; keep this.
- Several unused kernels/files were removed earlier (compute_loss, gemm_ffn,
  gemm_proj, gemm_bf16_gate_up, swiglu_forward, gemm_backward.metal,
  gemm_proj_trans_b.metal, gemm_bf16_naive) + CMake cleanup.

## 7. ALREADY-RULED-OUT (bisect results)

- **Not** the attention kernels (flash_attn_fwd, QKV gemms, reshape, rope):
  fault persists with the attention forced to CPU (`ATTN_CPU` env, since removed).
- **Not** the output-projection GEMM: persists with `SKIP_OUTPROJ`.
- **Not** the FFN (`w_gate`/`w_up` gemms, `fused_add_norm`, `fused_swiglu_gemm`):
  persists with `FFN_CPU` (forward FFN on CPU) — and `fused_swiglu_gemm` was
  fixed (5.1) yet the fault remains.
- **Not** memory capacity or data validity (section 4).
- The corruption is **not** caught by AddressSanitizer as a WRITE: an ASan build
  ran 29 steps catching only two CPU *reads* (the embedding-grad read — fixed,
  and a benign stream `memcpy` read). This implies the corrupting **write is
  GPU-side**, into Metal buffer memory that ASan cannot see.
- `MTL_DEBUG_LAYER=1` + `MTL_DEBUG_LAYER_ALLOW_UNSAFE_BUFFER_ACCESS=0` enabled
  but did not catch it (the OOB address is computed at runtime, not an
  encode-time buffer-range violation).
- Per-kernel Metal dispatch probe (all forward kernels, fresh buffers) → all
  passed. The fault only manifests with the training's **exact-sized,
  first-allocated buffers** (e.g. a `bfloat4`/`float4` tail load at a buffer
  boundary that is unmapped in the training layout but mapped in Metal-owned
  buffers).

## 8. REMAINING SUSPECTS (the actual task)

The step-0 forward GPU fault. After bisects, the only forward kernels that still
ran on GPU are:

- `embedding_forward.metal`
- `rms_norm_forward.metal` (scalar, looked in-bounds)
- `reshape_to_4d.metal` / `reshape_to_3d.metal`
- `rope_forward.metal`

All were audited and look in-bounds for the training shapes (dims=1024,
head_dim=64, all multiples of 4). **Yet a GPU page fault occurs at step 0.**
Likely one of these does a small vectorized/edge access that overruns an
exact-size buffer at the first allocation, OR there is a buffer-size mismatch
in its dispatch (kernel grid vs actual buffer length). `flash_attn_fwd.metal`
and `gemm_gqa.metal` also use `bfloat4` loads and are worth re-auditing for the
tail pattern even though bisects pointed away from attention.

Also consider: the corruption may be a *backward*-pass kernel's write OOB that
accumulates (the step-0 fault may be a separate benign read). Backward kernels
not yet fully audited for tail-vector writes: `rms_norm_backward.metal`
(lines 48-50, 106 — `bfloat4` load/store at `col_idx < dims`), `gemm_bf16.metal`
(C write), `gemm_gqa.metal` (line 156 write), `fused_backward_add_norm.metal`.

## 9. Recommended next steps (highest value first)

1. **Metal GPU frame capture (definitive).** The app supports
   `CAPTURE_METAL_TRACE=1` (`start_step_trace()` in `Trainer.cpp` at step 0),
   but it currently fails with "Capture layer is not inserted" — the app must be
   launched under Xcode with a Metal capture attachment (GPU Frame Capture →
   Metal, or `MTLCaptureManager` with a validated descriptor). The trace names
   the faulting kernel. Alternatively, set a `label` on each compute encoder
   (kernel name) in `MetalBridge.mm` dispatch functions; the command-buffer
   error's `userInfo` may then name the failing encoder.
2. **Audit `embedding_forward`, `rms_norm_forward`, `reshape_to_4d/3d`,
   `rope_forward` dispatch buffer sizes vs kernel grids** — specifically whether
   any kernel can access beyond `bytes` passed to `get_or_create_buffer`
   (e.g. `n_threads` rounded up, or `num_rows * dims` vs actual tensor bytes).
3. **Re-audit `flash_attn_fwd.metal` and `gemm_gqa.metal` bfloat4 loads** for the
   same tail pattern as 5.1 (guard only checks the start position).
4. **Add a guard/poison region around *all* Metal buffers** (including the
   NoCopy and PagedBuffer fast-path wrappers, not just size-pool copy buffers)
   and check after each `end_scope` to catch which kernel overruns. Previous
   guard check only covered size-pool buffers and was clean.
5. **Bisect the backward**: run with the backward forced to CPU per-stage
   (RMSNorm, FFN, Attention already have `use_gpu` branches) to see if the
   corruption disappears.

## 10. Key files

- `cpp/src/gpu_kernel/MetalBridge.mm` — buffer management + all dispatch. The
  buffer logic (5.3, 5.4) and the `[GPU] command buffer error` print in
  `end_scope` are here.
- `cpp/src/gpu_kernel/fused_swiglu_gemm.metal` — fixed tail loads (5.1).
- `cpp/src/gpu_kernel/embedding_forward.metal`, `cross_entropy.metal`,
  `rms_norm_forward.metal`, `rms_norm_backward.metal`, `reshape_4d.metal`,
  `reshape_3d.metal`, `rope_forward.metal`, `flash_attn_fwd.metal`,
  `gemm_gqa.metal`, `gemm_bf16.metal`, `gemm_bwd_weight.metal`.
- `cpp/src/Transformer.cpp` — `accumulate_embedding_grads` (5.2), forward/backward.
- `cpp/src/Attention.cpp`, `cpp/src/RMSNorm.cpp`, `cpp/src/Positional.cpp`.

## 11. Useful diagnostics already built (recreate if needed)

- ASan build: `cmake -S . -B build-asan -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"` then `cmake --build build-asan --target run_trainer`.
- Per-kernel probe: standalone ObjC++ that dispatches each kernel with fresh
  `newBufferWithLength` buffers at training shapes and checks `[commandBuffer error]`.
- GPU working-set probe: `[device recommendedMaxWorkingSetSize]` (= 498 GB).
- lldb: `lldb -b -o run -o "bt 15" -- ./build/run_trainer ...` → shows the GPU
  page fault (not a CPU backtrace; confirms GPU-side).

## 12. Repro commands to confirm a fix

```bash
cmake --build build --target run_trainer
# must run 100+ steps WITHOUT any "[GPU] command buffer error" or "bad weak table"
./build/run_trainer --batch_size 4 --max_steps 100 --checkpoint_interval 9999
# then the real target:
./build/run_trainer --batch_size 32 --max_steps 500 --checkpoint_interval 500
```
