import re

with open("cpp/src/gpu_kernel/MetalBridge.mm", "r") as f:
    code = f.read()

# 1. Update bufferCache definition
code = code.replace(
    "static std::unordered_map<const void *, id<MTLBuffer>> bufferCache;",
    "struct CachedBuffer {\n  id<MTLBuffer> buf;\n  bool is_persistent;\n};\nstatic std::unordered_map<const void *, CachedBuffer> bufferCache;"
)

# 2. Update get_or_create_buffer prototype
code = code.replace(
    "static id<MTLBuffer> get_or_create_buffer(const void *ptr, size_t bytes,\n                                          bool is_write = false);",
    "static id<MTLBuffer> get_or_create_buffer(const void *ptr, size_t bytes,\n                                          bool is_write = false, bool is_persistent = false);"
)

# 3. Update get_or_create_buffer implementation
code = code.replace(
    "static id<MTLBuffer> get_or_create_buffer(const void *ptr, size_t bytes,\n                                          bool is_write) {",
    "static id<MTLBuffer> get_or_create_buffer(const void *ptr, size_t bytes,\n                                          bool is_write, bool is_persistent) {"
)
code = code.replace(
    "    return it->second;",
    "    return it->second.buf;"
)
code = code.replace(
    "    bufferCache[ptr] = buf;",
    "    bufferCache[ptr] = {buf, is_persistent};"
)

# 4. Update commit_batch
code = code.replace(
    "  bufferCache.clear();\n}",
    "  for (auto it = bufferCache.begin(); it != bufferCache.end();) {\n    if (!it->second.is_persistent) {\n      it = bufferCache.erase(it);\n    } else {\n      ++it;\n    }\n  }\n}"
)

# 5. Make specific buffers persistent
# gemm_ffn
code = code.replace(
    "bufferB_gate = get_or_create_buffer(b_gate, bytesB);",
    "bufferB_gate = get_or_create_buffer(b_gate, bytesB, false, true);"
)
code = code.replace(
    "bufferB_up = get_or_create_buffer(b_up, bytesB);",
    "bufferB_up = get_or_create_buffer(b_up, bytesB, false, true);"
)

# gemm_proj
code = code.replace(
    "bufferB = get_or_create_buffer(b, bytesB);",
    "bufferB = get_or_create_buffer(b, bytesB, false, true);"
)

# rms_norm_forward
code = code.replace(
    "bufferW = get_or_create_buffer(weight, bytesW);",
    "bufferW = get_or_create_buffer(weight, bytesW, false, true);"
)

# rms_norm_backward
code = code.replace(
    "bufferWeight = get_or_create_buffer(weight, bytesW);",
    "bufferWeight = get_or_create_buffer(weight, bytesW, false, true);"
)
code = code.replace(
    "bufferGradWeight = \n      get_or_create_buffer(grad_weight, bytesW, true);",
    "bufferGradWeight = \n      get_or_create_buffer(grad_weight, bytesW, true, true);"
)
code = code.replace(
    "get_or_create_buffer(grad_weight, bytesW, true);",
    "get_or_create_buffer(grad_weight, bytesW, true, true);"
)

# rope_backward
code = code.replace(
    "bufferCos = get_or_create_buffer(cos_table, bytesTable);",
    "bufferCos = get_or_create_buffer(cos_table, bytesTable, false, true);"
)
code = code.replace(
    "bufferSin = get_or_create_buffer(sin_table, bytesTable);",
    "bufferSin = get_or_create_buffer(sin_table, bytesTable, false, true);"
)

# adamw_step
code = code.replace(
    "bufferParam = get_or_create_buffer(param, bytes, true);",
    "bufferParam = get_or_create_buffer(param, bytes, true, true);"
)
code = code.replace(
    "bufferGrad = get_or_create_buffer(grad, bytes);",
    "bufferGrad = get_or_create_buffer(grad, bytes, false, true);"
)
code = code.replace(
    "bufferM = get_or_create_buffer(m, bytes, true);",
    "bufferM = get_or_create_buffer(m, bytes, true, true);"
)
code = code.replace(
    "bufferV = get_or_create_buffer(v, bytes, true);",
    "bufferV = get_or_create_buffer(v, bytes, true, true);"
)

with open("cpp/src/gpu_kernel/MetalBridge.mm", "w") as f:
    f.write(code)

print("Done patching.")
