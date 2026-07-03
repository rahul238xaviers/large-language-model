#include <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdlib>
#include <iostream>
#include <malloc/_malloc.h>

id<MTLBuffer> createBufferWihBytesWithoutCopy(id<MTLDevice> device,
                                              float *host_ptr,
                                              size_t size_in_bytes) {
  id<MTLBuffer> buffer = [device
      newBufferWithBytesNoCopy:host_ptr
                        length:size_in_bytes
                       options:MTLResourceStorageModeShared
                   deallocator:^(void *pointer, NSUInteger length) {
                     fprintf(stderr, "A Buffer deallocated %p\n", pointer);
                     free(pointer);
                   }];
  ;

  return buffer;
}

int main() {
  @autoreleasepool {

    NSError *error = nil;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    NSURL *shaderURL = [NSURL fileURLWithPath:@"build/default.metallib"];

    id<MTLLibrary> library = [device newLibraryWithURL:shaderURL error:&error];

    if (library == nil) {

      NSString *localisedError = error.localizedDescription;
      const char *errorString = localisedError.UTF8String;
      fprintf(stderr, "Failed to load library: %s\n", errorString);
      return 1;
    }

    id<MTLFunction> vectoryMultiply =
        [library newFunctionWithName:@"vector_multiply"];

    if (vectoryMultiply == nil) {
      fprintf(stderr, "Failed to find vector multiply function");
      return 1;
    }

    id<MTLComputePipelineState> pipeLineState =
        [device newComputePipelineStateWithFunction:vectoryMultiply
                                              error:&error];

    if (pipeLineState == nil) {

      NSString *localisedError = error.localizedDescription;
      const char *errorString = localisedError.UTF8String;
      fprintf(stderr, "Failed to create compute pipeline state: %s\n",
              errorString);
      return 1;
    }

    const size_t num_elements = 1003520;
    const size_t bytes = sizeof(float) * num_elements;

    float *host_A = nullptr;
    float *host_B = nullptr;
    float *host_C = nullptr;

    int allocate_A = posix_memalign((void **)&host_A, 16384, bytes);
    int allocate_B = posix_memalign((void **)&host_B, 16384, bytes);
    int allocate_C = posix_memalign((void **)&host_C, 16384, bytes);

    if (allocate_A != 0 || allocate_B != 0 || allocate_C != 0) {
      fprintf(stderr, "Failed to allocate memory for host arrays");
      return 1;
    }

    for (int i = 0; i < num_elements; i++) {
      host_A[i] = (float)(i);
      host_B[i] = 2.0f;
      host_C[i] = 0.0f;
    }
    id<MTLBuffer> A_buffer =
        createBufferWihBytesWithoutCopy(device, host_A, bytes);

    id<MTLBuffer> B_buffer =
        createBufferWihBytesWithoutCopy(device, host_B, bytes);

    id<MTLBuffer> C_buffer =
        createBufferWihBytesWithoutCopy(device, host_C, bytes);

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeLineState];

    [encoder setBuffer:A_buffer offset:0 atIndex:0];
    [encoder setBuffer:B_buffer offset:0 atIndex:1];
    [encoder setBuffer:C_buffer offset:0 atIndex:2];

    uint32_t size = num_elements;
    [encoder setBytes:&size length:sizeof(size) atIndex:3];

    MTLSize gridSize = MTLSizeMake(num_elements, 1, 1);

    MTLSize threadGroupSize =
        MTLSizeMake(pipeLineState.threadExecutionWidth, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    for (size_t i = 0; i < num_elements; i++) {
      if (host_C[i] != host_A[i] * host_B[i]) {
        fprintf(stderr, "Error at index %zu: Expected %f, got %f\n", i,
                host_A[i] * host_B[i], host_C[i]);
        return 1;
      }
    }

    fprintf(stdout, "Verification Successful!!\n");
  }
}