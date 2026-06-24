cmake_minimum_required(VERSION 3.15)

# Set the macOS deployment target to match the host OS and Rust cargo target
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.5" CACHE STRING "macOS Deployment Target" FORCE)

project(LLM_Pipeline LANGUAGES CXX)

# Set the C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

