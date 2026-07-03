# 📖 Apple Silicon GPU Programming Bible (Metal & C++)

| Metadata | Description |
| :--- | :--- |
| **Author** | Principal GPU & HPC Architect |
| **Date** | July 2026 |
| **Status** | Approved & Audited (State-of-the-Art 2026) |
| **Target Architecture** | Apple Silicon UMA (M1-M5, A14-A19 GPU Families) |
| **Purpose** | Establish native C++ & MSL guidelines for high-performance LLM training backend |

---

> [!NOTE]
> **Architectural Vision Statement**: 
> This document serves as the absolute source of truth for engineering native GPU computation layers on Apple Silicon. It establishes strict constraints on memory layouts, compilation toolchains, and execution pipelines. Follow these guidelines to prevent bus congestion, hardware stalls, and memory-mapping faults.

---


## 🏛️ 1. The Core Architecture: Unified Memory (UMA)

Before writing code, it is essential to understand the physical layout of Apple Silicon (M-series chips). 

In traditional PC architectures, the CPU and GPU are separate chips with separate memory pools:
- **CPU** uses system RAM.
- **GPU** uses discrete Video RAM (VRAM) across a slow PCIe bus.
- **The Penalty**: To run a GPU task, the CPU must copy the data over the PCIe bus to VRAM, run the GPU kernel, and then copy the results back.

On **Apple Silicon**, the CPU and GPU are integrated onto the same silicon die and share a single pool of memory:
- **Zero-Copy**: The GPU can read and write to the exact same physical RAM bytes that the CPU allocated.
- No copying of data across buses is required. This is called **Zero-Copy Memory Mapping**.

### The Virtual Memory MMU Constraint
Even though memory is physically shared, the CPU and GPU still access it through a virtual sandbox for security and stability. The operating system's **MMU (Memory Management Unit)** maps memory at the granularity of **Virtual Memory Pages**.
- On Apple Silicon, a page is exactly **16 KB (16,384 bytes)**.
- To share C++ memory with the GPU without copying, the MMU must map entire pages.
- Therefore, any zero-copy memory allocation **must be aligned to a 16 KB boundary** and its **size must be a multiple of 16 KB**. If these rules are violated, the OS refuses to map the page and the GPU allocation fails.

---

## 📁 2. Host Languages: Objective-C++ (`.mm`) vs. `metal-cpp`

When writing C++ applications that talk to the Metal GPU driver, you have two choices for your host-side files:

### Approach A: Objective-C++ (`.mm` files) - *Classic & Standard*
- Combines standard C++ and Apple's Objective-C syntax in the same file.
- It is the standard way to write Metal host code because the Metal API was originally designed in Objective-C.
- Standard C++ code can call functions declared in these files using standard C linkage (`extern "C"`).

### Approach B: `metal-cpp` - *Modern Pure C++*
- A header-only C++ library provided by Apple.
- Allows you to write 100% pure C++ code (using standard classes like `NS::String` and `MTL::Device`) without using `.mm` files or Objective-C compilers.
- **Implementation Rule**: You must define `MTL_PRIVATE_IMPLEMENTATION` in exactly one translation unit (`.cpp` file) in your project to compile the Metal headers' private implementations.

---

## ⚙️ 3. The 7 Steps of the Metal GPU Pipeline

Every GPU program on macOS follows these 7 steps in the host code:

### Step 1: Locate the GPU (`MTL::Device` / `MTLDevice`)
We query the OS to get an interface to the physical GPU core:
```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
```

### Step 2: Create the Command Queue (`MTLCommandQueue`)
The GPU runs asynchronously. We submit command packets to a conveyor-belt queue:
```objc
id<MTLCommandQueue> queue = [device newCommandQueue];
```

### Step 3: Load & Compile the Shader (`MTLComputePipelineState`)
We compile the code inside our `.metal` file into binary microcode optimized for your specific GPU:
```objc
id<MTLLibrary> library = [device newDefaultLibrary];
id<MTLFunction> kernelFunc = [library newFunctionWithName:@"my_kernel_name"];
id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
```

### Step 4: Allocate Page-Aligned Host Memory
We allocate C++ heap memory aligned to 16 KB page boundaries using `posix_memalign`:
```cpp
void* host_ptr = nullptr;
posix_memalign(&host_ptr, 16384, size_in_bytes);
```

### Step 5: Wrap it in a Zero-Copy Buffer (`MTLBuffer`)
We tell the MMU to share this physical page of memory directly with the GPU:
```objc
id<MTLBuffer> gpu_buffer = [device newBufferWithBytesNoCopy:host_ptr
                                                     length:size_in_bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void* pointer, NSUInteger length) {
                                                    free(pointer); // Frees C++ memory when Metal releases the buffer
                                                }];
```

### Step 6: Encode the Instructions (`MTLComputeCommandEncoder`)
We write our execution commands (binding the buffers, specifying thread grid sizes) onto a command buffer:
```objc
id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
[encoder setComputePipelineState:pipelineState];
[encoder setBuffer:gpu_buffer offset:0 atIndex:0];
[encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
[encoder endEncoding]; // Close the encoder
```

### Step 7: Execute (`commit`)
We push the command buffer onto the queue conveyor belt:
```objc
[cmdBuffer commit];
[cmdBuffer waitUntilCompleted]; // Blocks CPU thread until GPU is done
```

---

## 🏎️ 4. 2026 Best Practices for High Performance

If you write a basic Metal pipeline, it will run, but it won't exploit Apple Silicon's full potential. Follow these modern performance rules:

### Rule 1: Never Hardcode the Threadgroup Size (SIMD size)
Apple Silicon GPUs have different execution widths (often called **warp size** or **SIMDgroup size**). 
- Older GPUs or efficiency blocks run **16 threads** per SIMDgroup.
- Newer high-performance GPUs run **32 threads** per SIMDgroup.
Always query the compiled pipeline at runtime rather than hardcoding it:
```objc
// Query the execution width dynamically
NSUInteger threadgroupSize = pipelineState.threadExecutionWidth; 
```

### Rule 2: Multi-Buffering (Double/Triple Buffering)
Calling `[cmdBuffer waitUntilCompleted]` puts the CPU thread to sleep, leaving it idle while the GPU is working. 
To achieve maximum throughput (e.g. during training), use **Double or Triple Buffering**:
- Allocate two (or three) sets of buffers (Buffer A and Buffer B).
- While the GPU is processing Buffer A, the CPU asynchronously prepares data in Buffer B.
- Use `addCompletedHandler:` on the command buffer to receive an asynchronous callback when the GPU is done, rather than blocking the thread.

### Rule 3: Memory Sub-allocation (Pooling)
Avoid creating separate zero-copy buffers for small variables (like loss scale, step count, or learning rate). Since the minimum size of a zero-copy buffer is **16 KB**, creating 10 small buffers wastes 160 KB and adds massive driver overhead.
- **Best Practice**: Allocate one large 16 KB block, pack your variables sequentially inside it, and pass it as a single `MTLBuffer`. In your shader, access each variable by its byte offset.

---

## 🚀 5. End-to-End Walkthrough: Element-Wise Float Multiplication

Let's design a program to multiply a large array of floats (e.g., $C_i = A_i \times B_i$).

### The GPU Shader (`multiply.metal`)
This runs in parallel on thousands of GPU threads. Each thread calculates a single index `gid`.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void float_multiply(
    device const float* A       [[buffer(0)]], // Input Array A
    device const float* B       [[buffer(1)]], // Input Array B
    device float*       C       [[buffer(2)]], // Output Array C
    device const uint&  size    [[buffer(3)]], // Total elements
    uint gid [[thread_position_in_grid]])      // Thread index
{
    // Ensure we don't read/write out of bounds
    if (gid < size) {
        C[gid] = A[gid] * B[gid];
    }
}
```

### The Host Code (`multiply_bridge.mm`)
This Objective-C++ file bridges C++ to Metal.

```objc
#include <cstdlib>
#include <iostream>
#import <Metal/Metal.h>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> pipelineState = nil;

extern "C" void gpu_initialize() {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
    
    NSError* error = nil;
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    
    // Load the compiled float_multiply function
    id<MTLFunction> multiplyFunc = [defaultLibrary newFunctionWithName:@"float_multiply"];
    pipelineState = [device newComputePipelineStateWithFunction:multiplyFunc error:&error];
    
    if (!pipelineState) {
        std::cerr << "Failed to compile pipeline state!" << std::endl;
    }
}

extern "C" void gpu_multiply(const float* host_A, const float* host_B, float* host_C, size_t size) {
    size_t bytes = size * sizeof(float);
    
    // Create zero-copy wrappers around the C++ page-aligned pointers
    id<MTLBuffer> bufferA = [device newBufferWithBytesNoCopy:(void*)host_A length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bufferB = [device newBufferWithBytesNoCopy:(void*)host_B length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bufferC = [device newBufferWithBytesNoCopy:(void*)host_C length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];
    
    uint sz = static_cast<uint>(size);
    [encoder setBytes:&sz length:sizeof(uint) atIndex:3];
    
    // 2026 Best Practice: Dynamic thread execution width query
    NSUInteger maxThreads = pipelineState.maxTotalThreadsPerThreadgroup;
    NSUInteger executionWidth = pipelineState.threadExecutionWidth; 
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(executionWidth, 1, 1); // Optimal warp size alignment
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}
```

---

## 🛠️ 6. Compilation Walkthrough

To compile this pipeline under your custom `build/` directory workflow:

1. **Compile the Metal Shaders**:
   The OS compiles `.metal` files into a `.metallib` (Metal library) binary:
   ```bash
   xcrun -sdk macosx metal -c multiply.metal -o build/multiply.air
   xcrun -sdk macosx metallib build/multiply.air -o build/default.metallib
   ```

2. **Compile the Host C++ & Objective-C++ Code**:
   We link the Foundation and Metal frameworks:
   ```bash
   clang++ -std=c++20 -framework Metal -framework Foundation -I. \
       multiply_bridge.mm main.cpp -o build/multiply.o
   ```
