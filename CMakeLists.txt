cmake_minimum_required(VERSION 3.15)

# Set the macOS deployment target to match the host OS and Rust cargo target
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.5" CACHE STRING "macOS Deployment Target" FORCE)

project(LLM_Pipeline LANGUAGES CXX)

# Set the C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Release build type by default if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Set optimization flags for high performance CPU execution
if(MSVC)
  add_compile_options(/O2 /Gy)
else()
  add_compile_options(-O3)
endif()

# Export compile_commands.json for Clangd/IDE LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add Homebrew to search path on macOS
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

# Find Arrow and Parquet packages
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Add tokenizers-cpp library target (disable SentencePiece to reduce build complexity)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "Disable SentencePiece" FORCE)
add_subdirectory(cpp/tokenizers-cpp)

# Define our library target
add_library(data_ingestion
    cpp/src/DataIngestion.cpp
    cpp/src/Tensor.cpp
    cpp/src/RMSNorm.cpp
    cpp/src/Activations.cpp
    cpp/src/Positional.cpp
    cpp/src/Attention.cpp
    cpp/src/Transformer.cpp
    cpp/src/Loss.cpp
    cpp/src/Optimizer.cpp
    cpp/src/Trainer.cpp
    cpp/src/gpu_kernel/MetalBridge.mm
)

# Set include directories
target_include_directories(data_ingestion PUBLIC
    cpp/include
)

# Link against Arrow, Parquet, and tokenizers-cpp static libraries
target_link_libraries(data_ingestion PUBLIC
    arrow_shared
    parquet_shared
    tokenizers_cpp
    "-framework Accelerate"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-framework Foundation"
)

# Define our test executable
add_executable(test_data_ingestion
    cpp/tests/test_data_ingestion.cpp
)

# Link against our library
target_link_libraries(test_data_ingestion PUBLIC
    data_ingestion
)

# Define our mathematical operations test executable
add_executable(test_math_ops
    cpp/tests/test_math_ops.cpp
)
# Link it against our library
target_link_libraries(test_math_ops PUBLIC
    data_ingestion
)

# Define our RMSNorm test executable
add_executable(test_rmsnorm
    cpp/tests/test_rmsnorm.cpp
)
# Link it against our library
target_link_libraries(test_rmsnorm PUBLIC
    data_ingestion
)

# Define our Activations test executable
add_executable(test_activations
    cpp/tests/test_activations.cpp
)
# Link it against our library
target_link_libraries(test_activations PUBLIC
    data_ingestion
)

# Define our RoPE test executable
add_executable(test_rope
    cpp/tests/test_rope.cpp
)
# Link it against our library
target_link_libraries(test_rope PUBLIC
    data_ingestion
)

# Define our Attention test executable
add_executable(test_attention
    cpp/tests/test_attention.cpp
)
# Link it against our library
target_link_libraries(test_attention PUBLIC
    data_ingestion
)

# Define our Transformer test executable
add_executable(test_transformer
    cpp/tests/test_transformer.cpp
)
# Link it against our library
target_link_libraries(test_transformer PUBLIC
    data_ingestion
)

# Define our Loss test executable
add_executable(test_loss
    cpp/tests/test_loss.cpp
)
# Link it against our library
target_link_libraries(test_loss PUBLIC
    data_ingestion
)

# Define our complete backprop test executable
add_executable(test_backward
    cpp/tests/test_backward.cpp
)
# Link it against our library
target_link_libraries(test_backward PUBLIC
    data_ingestion
)

# Define our Trainer test executable
add_executable(test_trainer
    cpp/tests/test_trainer.cpp
)
# Link it against our library
target_link_libraries(test_trainer PUBLIC
    data_ingestion
)

# 1. Find the xcrun tool on macOS
find_program(XCRUN xcrun)

# 2. Configure maximum sequence length for the GQA shader
set(MAX_SEQ_LEN 2048 CACHE STRING "Maximum sequence length for GQA attention kernel")

# 3. Add custom commands to compile our shaders
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/default.metallib
    COMMAND ${XCRUN} -sdk macosx metal -c
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/gemm_ffn.metal
            -o ${CMAKE_BINARY_DIR}/gemm_ffn.air
    COMMAND ${XCRUN} -sdk macosx metal -c
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/gemm_proj.metal
            -o ${CMAKE_BINARY_DIR}/gemm_proj.air
    COMMAND ${XCRUN} -sdk macosx metal -c
            -DMAX_SEQ_LEN=${MAX_SEQ_LEN}
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/gemm_gqa.metal
            -o ${CMAKE_BINARY_DIR}/gemm_gqa.air
    COMMAND ${XCRUN} -sdk macosx metal -c
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/rms_norm_forward.metal
            -o ${CMAKE_BINARY_DIR}/rms_norm_forward.air
    COMMAND ${XCRUN} -sdk macosx metal -c
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/rms_norm_backward.metal
            -o ${CMAKE_BINARY_DIR}/rms_norm_backward.air
    COMMAND ${XCRUN} -sdk macosx metallib
            ${CMAKE_BINARY_DIR}/gemm_ffn.air
            ${CMAKE_BINARY_DIR}/gemm_proj.air
            ${CMAKE_BINARY_DIR}/gemm_gqa.air
            ${CMAKE_BINARY_DIR}/rms_norm_forward.air
            ${CMAKE_BINARY_DIR}/rms_norm_backward.air
            -o ${CMAKE_BINARY_DIR}/default.metallib
    DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/gemm_ffn.metal
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/gemm_proj.metal
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/gemm_gqa.metal
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/rms_norm_forward.metal
            ${CMAKE_SOURCE_DIR}/cpp/src/gpu_kernel/rms_norm_backward.metal
)
# 3. Add default.metallib to the target files so CMake knows it must build it
add_custom_target(compile_shaders DEPENDS ${CMAKE_BINARY_DIR}/default.metallib)
add_dependencies(data_ingestion compile_shaders)

# 4. Define our GPU kernels verification test executable
add_executable(test_gpu_kernels
    cpp/tests/test_gpu_kernels.cpp
)
# Link it against our library
target_link_libraries(test_gpu_kernels PUBLIC
    data_ingestion
)


