#include <metal_stdlib>

using namespace metal;

kernel void vector_multiply(device const float* A[[buffer(0)]], device const float* B[[buffer(1)]], 
device float* C [[buffer(2)]],device const uint& size[[buffer(3)]], 
unsigned int gid [[thread_position_in_grid]]){

    if(gid < size){
        C[gid] = A [gid] * B [gid];
    }

}