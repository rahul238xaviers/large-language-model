/**
 * @file MetalBridge.mm
 * @brief Implementation of the C++ to Metal GPU runtime bridge.
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Orchestrates GPU memory buffers creation, manages pipeline compute states,
 * and submits execution dispatches to Apple Silicon hardware (using MPS matrix cores).
 */

#include "gpu_kernel/MetalBridge.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <simd/simd.h>
#include <cstddef>
#include <iostream>

namespace metal_bridge {

// device (MTLDevice): Represents the physical GPU hardware.
// Think of it as the parent manager of all GPU memory and pipeline allocations.
static id<MTLDevice> device = nil;

// commandQueue (MTLCommandQueue): A thread-safe FIFO (first-in-first-out)
// queue. This is the conveyor belt that carries instruction packets (command
// buffers) from CPU to GPU.
static id<MTLCommandQueue> commandQueue = nil;

// pipelineState (MTLComputePipelineState): The compiled microcode for our GPU
// kernel. This contains the actual GPU machine instructions for the vector_add
// function.
static id<MTLComputePipelineState> pipelineState = nil;

// pipelineStateFFN (MTLComputePipelineState): The compiled microcode for our tiled FFN GEMM.
static id<MTLComputePipelineState> pipelineStateFFN = nil;

static bool initialized = false;

// Profiling statistics definitions
double accum_gpu_time_ms = 0.0;
double accum_cpu_time_ms = 0.0;
size_t count_gpu_calls = 0;
size_t count_cpu_calls = 0;

/**
 * @brief Check if the GPU device and compile pipeline states are fully loaded and operational.
 *
 * @return true If available.
 * @return false Otherwise.
 */
bool is_available() {
  return (device != nil) && (pipelineStateFFN != nil);
}

/**
 * @brief Reset accumulated GPU and CPU execution profiling timers and counts to 0.
 */
void reset_profile_stats() {
  accum_gpu_time_ms = 0.0;
  accum_cpu_time_ms = 0.0;
  count_gpu_calls = 0;
  count_cpu_calls = 0;
}

/**
 * @brief Searches for the Apple GPU device and compiles the Metal shader libraries.
 */
void initialize() {
  if (initialized) return;
  // 1. MTLCreateSystemDefaultDevice()
  // WHAT: A system-level function that searches the Apple Silicon chip for the
  // GPU core block
  //       and initializes a driver interface for it.
  // WHY: This interface is required to create all other Metal objects (queues,
  // buffers, libraries).
  device = MTLCreateSystemDefaultDevice();
  if (!device) {
    std::cerr << "Failed to find a Metal GPU device!" << std::endl;
    return;
  }
  // [device.name UTF8String] calls the 'name' property on the device object
  // (which returns an NSString like @"Apple M5") and converts it to a standard
  // C++ char* pointer.
  std::cout << "Metal GPU initialized: " << [device.name UTF8String]
            << std::endl;

  // 2. [device newCommandQueue]
  // WHAT: A method called on the device object that allocates a command queue
  // structure. WHY: The GPU is asynchronous. We cannot send instructions
  // directly to the GPU cores.
  //      We must submit them to this queue, which handles the hardware
  //      execution.
  // C++ equivalent: commandQueue = device->newCommandQueue();
  commandQueue = [device newCommandQueue];

  NSError *error = nil;

  // 3. [device newDefaultLibrary]
  // WHAT: Searches the application's running bundle/directory for a
  // pre-compiled
  //       shader library file named 'default.metallib' and loads it into
  //       memory.
  // WHY: A shader library contains all of our compiled GPU functions. We must
  // load it
  //      into RAM first so that we can look inside it and locate the specific
  //      functions we want.
  id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];

  // 4. [device newLibraryWithURL:error:]
  // WHAT: Loads the compiled binary library from a URL path in the filesystem
  // instead of a plain string. WHY: Apple deprecated the string-based
  // 'newLibraryWithFile' method in macOS 13 to enforce the use of URLs,
  //      which are more secure and support modern sandboxing and network paths
  //      uniformly.
  // PARAMETERS:
  // - libraryURL: An NSURL object representing the file path
  // 'default.metallib'.
  // - error: Pointer to our NSError object to capture compilation/loading
  // diagnostics. C++ Translation: NSURL* libraryURL = [NSURL
  // fileURLWithPath:@"default.metallib"]; defaultLibrary =
  // device->newLibraryWithURL(libraryURL, &error);
  if (!defaultLibrary) {
    NSURL *libraryURL = [NSURL fileURLWithPath:@"default.metallib"];
    defaultLibrary = [device newLibraryWithURL:libraryURL error:&error];
  }
  if (!defaultLibrary) {
    std::cerr << "Failed to load default.metallib! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }

  // 5. [defaultLibrary newFunctionWithName:@"vector_add"]
  // WHAT: Parses the loaded binary library file, finds the function named
  // "vector_add",
  //       and extracts its entry-point information.
  // WHY: A library is a collection of many different GPU functions. We must
  // specify
  //      exactly which function we want to select before we can compile it.
  id<MTLFunction> kernelFunction =
      [defaultLibrary newFunctionWithName:@"vector_add"];
  if (!kernelFunction) {
    std::cerr << "Failed to find kernel function 'vector_add' in library!"
              << std::endl;
    return;
  }

  // 6. [device newComputePipelineStateWithFunction:error:]
  // WHAT: The critical compilation step. Takes the intermediate shader code
  // (from the library)
  //       and compiles it into the exact machine instructions (binary) for your
  //       specific M5 GPU cores.
  // WHY: A GPU cannot run source code or generic intermediate representations.
  // It must compile
  //      them into microcode for its specific execution units.
  pipelineState = [device newComputePipelineStateWithFunction:kernelFunction
                                                        error:&error];
  if (!pipelineState) {
    std::cerr << "Failed to compile pipeline state! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'vector_add'!"
            << std::endl;

  // 7. [defaultLibrary newFunctionWithName:@"gemm_ffn"]
  // WHAT: Finds the FFN GEMM kernel in our compiled shader library.
  id<MTLFunction> ffnFunction =
      [defaultLibrary newFunctionWithName:@"gemm_ffn"];
  if (!ffnFunction) {
    std::cerr << "Failed to find kernel function 'gemm_ffn' in library!"
              << std::endl;
    return;
  }

  // 8. Compile the FFN kernel function into pipelineStateFFN
  pipelineStateFFN = [device newComputePipelineStateWithFunction:ffnFunction
                                                           error:&error];
  if (!pipelineStateFFN) {
    std::cerr << "Failed to compile pipeline state for 'gemm_ffn'! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'gemm_ffn'!"
            << std::endl;
  initialized = true;
}

void vector_add(const float *a, const float *b, float *c, size_t size) {
  // WHAT: A safety guard checking if the vector size is equal to zero.
  // WHY: If size is 0, the byte count is 0, which causes
  // newBufferWithBytesNoCopy to crash because Metal cannot map an empty block.
  //      It also prevents us from trying to launch 0 threads on the GPU.
  if (size == 0)
    return;

  // WHAT: Calculates the total memory size of our float arrays in bytes (size *
  // 4 bytes). WHY: Metal's memory allocation and mapping APIs operate at the
  // hardware level in raw bytes,
  //      so we must calculate the exact memory footprint of our float arrays.
  size_t bytes = size * sizeof(float);

  // WHAT: Creates an id<MTLBuffer> (GPU buffer object) called bufferA pointing
  // to the CPU memory address (void*)a. WHY: By using newBufferWithBytesNoCopy,
  // we exploit Apple Silicon's Unified Memory. The GPU can read our C++ array
  // 'a'
  //      directly over the memory bus. This is completely zero-copy: no
  //      allocation of new memory blocks and no data copying happens.
  // PARAMETERS:
  // - (void*)a: The CPU memory address where our float array starts in RAM.
  // - length:bytes: The boundary of the memory block in bytes.
  // - options:MTLResourceStorageModeShared: Tells the hardware that both the
  // CPU and GPU can read/write this memory concurrently.
  // - deallocator:nil: Tells Metal that C++ owns this memory, so do not try to
  // free/delete it when this buffer object is destroyed. C++ Translation:
  // MTLBuffer* bufferA = device->newBufferWithBytesNoCopy(a, bytes,
  // MTLResourceStorageModeShared, nil);
  id<MTLBuffer> bufferA =
      [device newBufferWithBytesNoCopy:(void *)a
                                length:bytes
                               options:MTLResourceStorageModeShared
                           deallocator:nil];

  // WHAT: Creates a GPU buffer object called bufferB pointing to CPU address
  // (void*)b. WHY: To give the GPU direct, zero-copy access to read the C++
  // array 'b' over the memory bus without copying a single byte. C++
  // Translation: MTLBuffer* bufferB = device->newBufferWithBytesNoCopy(b,
  // bytes, MTLResourceStorageModeShared, nil);
  id<MTLBuffer> bufferB =
      [device newBufferWithBytesNoCopy:(void *)b
                                length:bytes
                               options:MTLResourceStorageModeShared
                           deallocator:nil];

  // WHAT: Creates a GPU buffer object called bufferC pointing to CPU address
  // (void*)c. WHY: To give the GPU direct, zero-copy access to write results
  // into the C++ array 'c' memory directly over the memory bus. C++
  // Translation: MTLBuffer* bufferC = device->newBufferWithBytesNoCopy(c,
  // bytes, MTLResourceStorageModeShared, nil);
  id<MTLBuffer> bufferC =
      [device newBufferWithBytesNoCopy:(void *)c
                                length:bytes
                               options:MTLResourceStorageModeShared
                           deallocator:nil];

  // WHAT: Checks if any of the three buffer pointers returned are nil (null).
  // WHY: If virtual memory mapping fails, these methods return nil. We must
  // exit immediately to prevent the CPU
  //      from executing commands on null references, which would cause a GPU
  //      crash.
  if (!bufferA || !bufferB || !bufferC) {
    std::cerr << "Failed to allocate Metal buffers!" << std::endl;
    return;
  }

  // WHAT: Requests a new empty MTLCommandBuffer object (cmdBuffer) from our
  // commandQueue. WHY: Because communicating with the GPU has high latency, we
  // cannot send instructions one-by-one.
  //      We must write all instructions onto this command buffer (our
  //      instruction sheet) and send it as a single job package.
  // C++ Translation: MTLCommandBuffer* cmdBuffer =
  // commandQueue->commandBuffer();
  id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];

  // WHAT: Creates a MTLComputeCommandEncoder object (computeEncoder) bound to
  // write into 'cmdBuffer'. WHY: The command buffer is a raw binary container.
  // We cannot write instructions to it directly.
  //      The encoder provides the API (the pen) to write structured GPU
  //      commands.
  // C++ Translation: MTLComputeCommandEncoder* computeEncoder =
  // cmdBuffer->computeCommandEncoder();
  id<MTLComputeCommandEncoder> computeEncoder =
      [cmdBuffer computeCommandEncoder];

  // WHAT: Writes the instruction: "The GPU should load our compiled
  // 'vector_add' shader program." WHY: The GPU cores can run many different
  // shaders. We must tell the encoder which specific set of
  //      microcode instructions to load before we start binding data or
  //      launching threads.
  // C++ Translation: computeEncoder->setComputePipelineState(pipelineState);
  [computeEncoder setComputePipelineState:pipelineState];

  // WHAT: Writes the instruction: "Link 'bufferA' to shader parameter slot 0."
  // WHY: Connects our C++ memory address to the [[buffer(0)]] annotation in our
  // shader (vector_add.metal).
  //      The offset:0 tells the GPU to start reading from the very beginning of
  //      the buffer.
  // C++ Translation: computeEncoder->setBuffer(bufferA, 0, 0);
  [computeEncoder setBuffer:bufferA offset:0 atIndex:0];

  // WHAT: Writes the instruction: "Link 'bufferB' to shader parameter slot 1."
  // WHY: Connects our C++ memory address to the [[buffer(1)]] annotation in our
  // shader. C++ Translation: computeEncoder->setBuffer(bufferB, 0, 1);
  [computeEncoder setBuffer:bufferB offset:0 atIndex:1];

  // WHAT: Writes the instruction: "Link 'bufferC' to shader parameter slot 2."
  // WHY: Connects our C++ memory address to the [[buffer(2)]] annotation in our
  // shader. C++ Translation: computeEncoder->setBuffer(bufferC, 0, 2);
  [computeEncoder setBuffer:bufferC offset:0 atIndex:2];

  // WHAT: Casts our C++ size (64-bit size_t) to a 32-bit unsigned integer
  // (uint).
  // WHY: To match the 32-bit 'uint' type expected by our Metal shader parameter
  // [[buffer(3)]].
  uint sz = static_cast<uint>(size);

  // WHAT: Writes the value of 'sz' directly into the command buffer at index 3.
  // WHY: Connects the size value to the '[[buffer(3)]]' parameter in the shader
  // without allocating a buffer. C++ Translation: computeEncoder->setBytes(&sz,
  // sizeof(uint), 3);
  [computeEncoder setBytes:&sz length:sizeof(uint) atIndex:3];

  // WHAT: Creates a 3D size structure (MTLSize) defining the width of our
  // thread block (Threadgroup). WHY: We query 'maxTotalThreadsPerThreadgroup'
  // from the compiled pipeline. This asks the GPU:
  //      "What is the maximum number of threads a single core can execute for
  //      this program?" (usually 1024). We set the width to this maximum to
  //      maximize GPU core ALU occupancy for this specific program. Since our
  //      vector is 1-dimensional, height and depth are set to 1.
  MTLSize threadGroupSize =
      MTLSizeMake(pipelineState.maxTotalThreadsPerThreadgroup, 1, 1);

  // WHAT: Creates a 3D size structure (MTLSize) representing the total threads
  // to run (equal to size).
  // WHY: To tell the GPU the total number of elements that must be calculated
  // in parallel. C++ Translation: MTLSize gridSize = MTLSizeMake(size, 1, 1);
  MTLSize gridSize = MTLSizeMake(size, 1, 1);

  // WHAT: Writes the instruction: "Launch gridSize threads, partitioned into
  // threadGroupSize blocks." WHY: Triggers the GPU hardware scheduling block to
  // map our 2,048 threads onto the GPU cores. C++ Translation:
  // computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  [computeEncoder dispatchThreads:gridSize
            threadsPerThreadgroup:threadGroupSize];

  // WHAT: Closes the encoder (the pen) writing session.
  // WHY: Metal requires us to explicitly close the encoder before we can submit
  // the command buffer. C++ Translation: computeEncoder->endEncoding();
  [computeEncoder endEncoding];

  // WHAT: Submits the completed command buffer packet to the conveyor belt
  // (commandQueue). WHY: This triggers the GPU hardware scheduler to pick up
  // the packet and start executing. C++ Translation: cmdBuffer->commit();
  [cmdBuffer commit];

  // WHAT: Puts the C++ thread to sleep until the GPU signals that the execution
  // has finished. WHY: To ensure the GPU has written all computed values back
  // to RAM before our C++ code reads them. C++ Translation:
  // cmdBuffer->waitUntilCompleted();
  [cmdBuffer waitUntilCompleted];
}

void gemm_ffn(const float *a, const float *b, float *c, size_t M, size_t N, size_t K) {
  // WHAT: Safety guard checking if any dimension is zero.
  // WHY: A matrix multiplication with size 0 has no work to do, and would cause division/allocation errors.
  if (M == 0 || N == 0 || K == 0) return;

  // WHAT: Calculates the byte sizes of our three matrices.
  // WHY: Zero-copy buffer allocation operates on raw byte counts.
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  // WHAT: Creates a zero-copy pointer to CPU Matrix A.
  // WHY: Unified memory allows the GPU to read A directly from CPU RAM without copying.
  id<MTLBuffer> bufferA = [device newBufferWithBytesNoCopy:(void *)a length:bytesA options:MTLResourceStorageModeShared deallocator:nil];

  // WHAT: Creates a zero-copy pointer to CPU Matrix B.
  // WHY: Unified memory allows the GPU to read B directly from CPU RAM without copying.
  id<MTLBuffer> bufferB = [device newBufferWithBytesNoCopy:(void *)b length:bytesB options:MTLResourceStorageModeShared deallocator:nil];

  // WHAT: Creates a zero-copy pointer to CPU Matrix C.
  // WHY: Unified memory allows the GPU to write results directly into CPU RAM.
  id<MTLBuffer> bufferC = [device newBufferWithBytesNoCopy:(void *)c length:bytesC options:MTLResourceStorageModeShared deallocator:nil];

  // WHAT: Validates all buffer allocations succeeded.
  // WHY: To prevent accessing null references, which crashes the GPU.
  if (!bufferA || !bufferB || !bufferC) {
    std::cerr << "Failed to allocate Metal GEMM buffers!" << std::endl;
    return;
  }

  // WHAT: Execute MPS Matrix Multiplication within an autorelease pool.
  // WHY: Metal Performance Shaders are objective-C classes that allocate temporary wrappers. Wrapping
  //      them in @autoreleasepool ensures they are immediately freed, avoiding memory growth.
  @autoreleasepool {
    // WHAT: Creates a command buffer packet (cmdBuffer) for the GPU.
    // WHY: To bundle all GEMM configuration commands into a single transmission.
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];

    // WHAT: Defines the shape and row stride of Matrix A.
    // WHY: Tells MPS the row-major format and byte offsets for reading input A.
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                      columns:K
                                                                     rowBytes:K * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
    
    // WHAT: Defines the shape and row stride of Matrix B.
    // WHY: Tells MPS the row-major format and byte offsets for reading input B.
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                      columns:N
                                                                     rowBytes:N * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
    
    // WHAT: Defines the shape and row stride of Matrix C.
    // WHY: Tells MPS where and how to write the output matrix C.
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                      columns:N
                                                                     rowBytes:N * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];

    // WHAT: Instantiates an MPSMatrix wrapper around bufferA.
    // WHY: Connects the raw Metal buffer data to the MPS matrix math object.
    MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];

    // WHAT: Instantiates an MPSMatrix wrapper around bufferB.
    // WHY: Connects the raw Metal buffer data to the MPS matrix math object.
    MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];

    // WHAT: Instantiates an MPSMatrix wrapper around bufferC.
    // WHY: Connects the raw Metal buffer data to the MPS matrix math object.
    MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

    // WHAT: Instantiates the high-performance MPS Matrix Multiplication kernel.
    // WHY: Instructs Apple's hardware matrix engines to perform C = A @ B.
    MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                        transposeLeft:NO
                                                                       transposeRight:NO
                                                                           resultRows:M
                                                                        resultColumns:N
                                                                      interiorColumns:K
                                                                                alpha:1.0f
                                                                                 beta:0.0f];

    // WHAT: Encodes the matrix multiplication execution command into the command buffer.
    // WHY: Queues the task to be run by the GPU shader cores.
    [matmul encodeToCommandBuffer:cmdBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];

    // WHAT: Submits the command buffer packet to the commandQueue conveyor belt.
    // WHY: Triggers the GPU hardware to execute the tasks.
    [cmdBuffer commit];

    // WHAT: Blocks the C++ thread until the GPU finishes calculating the GEMM.
    // WHY: Guarantees the output matrix C contains the results before we read it on the CPU.
    [cmdBuffer waitUntilCompleted];
  }
}

} // namespace metal_bridge
