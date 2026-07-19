/**
 * @file Checkpoint.cpp
 * @brief Implementation of safetensors-compatible checkpoint serialization and
 * deserialization, including Python/MLX compatibility adapters.
 */

#include "Checkpoint.hpp"
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cmath>

struct SavedTensor {
  std::string name;
  std::vector<size_t> shape;
  const float *data = nullptr;
  size_t size = 0;
  std::vector<float> owned_data;
  std::string dtype = "F32";
  const char *raw_data = nullptr;
  size_t raw_size_bytes = 0;
};

// BF16 to F32 conversion
static float bf16_to_f32(uint16_t val) {
  uint32_t val32 = ((uint32_t)val) << 16;
  float f;
  std::memcpy(&f, &val32, sizeof(float));
  return f;
}

// F16 to F32 conversion (fully compliant half to float)
static float f16_to_f32(uint16_t val) {
  uint32_t sign = (val & 0x8000) << 16;
  uint32_t exp = (val & 0x7C00) >> 10;
  uint32_t mant = val & 0x03FF;
  
  if (exp == 0) {
    if (mant == 0) {
      uint32_t val32 = sign;
      float f;
      std::memcpy(&f, &val32, sizeof(float));
      return f;
    } else {
      while ((mant & 0x0400) == 0) {
        mant <<= 1;
        exp--;
      }
      exp++;
      mant &= ~0x0400;
    }
  } else if (exp == 31) {
    uint32_t val32 = sign | 0x7F800000 | (mant << 13);
    float f;
    std::memcpy(&f, &val32, sizeof(float));
    return f;
  }
  
  exp = exp - 15 + 127;
  uint32_t val32 = sign | (exp << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &val32, sizeof(float));
  return f;
}

// Transpose helper
static void transpose_matrix(const float *src, float *dst, size_t src_rows, size_t src_cols) {
  for (size_t r = 0; r < src_rows; ++r) {
    for (size_t c = 0; c < src_cols; ++c) {
      dst[c * src_rows + r] = src[r * src_cols + c];
    }
  }
}

// Transpose to std::vector helper
static std::vector<float> transpose_to_vector(const Tensor &t) {
  size_t rows = t.shape()[0];
  size_t cols = t.shape()[1];
  std::vector<float> dst(rows * cols);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      dst[c * rows + r] = t(r, c);
    }
  }
  return dst;
}

// Fuses and transposes Wq, Wk, Wv into a single Python wqkv weight block
static std::vector<float> fuse_and_transpose_wqkv(const Tensor &Wq, const Tensor &Wk, const Tensor &Wv) {
  size_t hidden_dim = Wq.shape()[0];
  size_t q_features = Wq.shape()[1];
  size_t kv_features = Wk.shape()[1];
  
  std::vector<float> fused((q_features + 2 * kv_features) * hidden_dim);
  
  // Wq_T shape: [q_features, hidden_dim]
  for (size_t r = 0; r < hidden_dim; ++r) {
    for (size_t c = 0; c < q_features; ++c) {
      fused[c * hidden_dim + r] = Wq(r, c);
    }
  }
  
  // Wk_T shape: [kv_features, hidden_dim]
  size_t offset1 = q_features * hidden_dim;
  for (size_t r = 0; r < hidden_dim; ++r) {
    for (size_t c = 0; c < kv_features; ++c) {
      fused[offset1 + c * hidden_dim + r] = Wk(r, c);
    }
  }
  
  // Wv_T shape: [kv_features, hidden_dim]
  size_t offset2 = (q_features + kv_features) * hidden_dim;
  for (size_t r = 0; r < hidden_dim; ++r) {
    for (size_t c = 0; c < kv_features; ++c) {
      fused[offset2 + c * hidden_dim + r] = Wv(r, c);
    }
  }
  
  return fused;
}

// Fuses and transposes gate and up projections into a single Python w12 weight block
static std::vector<float> fuse_and_transpose_w12(const Tensor &w_gate, const Tensor &w_up) {
  size_t hidden_dim = w_gate.shape()[0];
  size_t intermediate_dim = w_gate.shape()[1];
  
  std::vector<float> fused(2 * intermediate_dim * hidden_dim);
  
  // w_gate_T shape: [intermediate_dim, hidden_dim]
  for (size_t r = 0; r < hidden_dim; ++r) {
    for (size_t c = 0; c < intermediate_dim; ++c) {
      fused[c * hidden_dim + r] = w_gate(r, c);
    }
  }
  
  // w_up_T shape: [intermediate_dim, hidden_dim]
  size_t offset = intermediate_dim * hidden_dim;
  for (size_t r = 0; r < hidden_dim; ++r) {
    for (size_t c = 0; c < intermediate_dim; ++c) {
      fused[offset + c * hidden_dim + r] = w_up(r, c);
    }
  }
  
  return fused;
}

static std::vector<std::pair<std::string, Tensor *>>
get_model_tensors(Transformer &model) {
  std::vector<std::pair<std::string, Tensor *>> tensors;
  tensors.push_back({"token_embeddings", &model.token_embeddings()});
  tensors.push_back({"output_projection", &model.output_projection()});
  tensors.push_back({"final_norm.weight", &model.final_norm().weight()});

  for (size_t l = 0; l < model.layers().size(); ++l) {
    std::string prefix = "layers." + std::to_string(l) + ".";
    auto &layer = model.layers()[l];
    tensors.push_back({prefix + "attn_norm.weight", &layer.attn_norm.weight()});
    tensors.push_back({prefix + "attn.Wq", &layer.attn.Wq()});
    tensors.push_back({prefix + "attn.Wk", &layer.attn.Wk()});
    tensors.push_back({prefix + "attn.Wv", &layer.attn.Wv()});
    tensors.push_back({prefix + "attn.Wo", &layer.attn.Wo()});
    tensors.push_back({prefix + "ffn_norm.weight", &layer.ffn_norm.weight()});
    tensors.push_back({prefix + "w_gate", &layer.w_gate});
    tensors.push_back({prefix + "w_up", &layer.w_up});
    tensors.push_back({prefix + "w_down", &layer.w_down});
  }
  return tensors;
}

static std::vector<std::string> get_optimizer_param_names(Transformer &model) {
  std::vector<std::string> names;
  names.push_back("token_embeddings");
  names.push_back("output_projection");
  for (size_t l = 0; l < model.layers().size(); ++l) {
    std::string prefix = "layers." + std::to_string(l) + ".";
    names.push_back(prefix + "w_gate");
    names.push_back(prefix + "w_up");
    names.push_back(prefix + "w_down");
    names.push_back(prefix + "attn.Wq");
    names.push_back(prefix + "attn.Wk");
    names.push_back(prefix + "attn.Wv");
    names.push_back(prefix + "attn.Wo");
  }
  return names;
}

static std::string build_safetensors_json(
    const std::vector<std::pair<std::string, Tensor *>> &tensors,
    std::vector<size_t> &offsets) {
  std::string json = "{";
  json += "\"__metadata__\":{\"format\":\"pt\"}";

  size_t current_offset = 0;
  offsets.clear();
  offsets.push_back(0);

  for (const auto &p : tensors) {
    const std::string &name = p.first;
    const Tensor *t = p.second;
    size_t size_bytes = t->size() * sizeof(float);
    size_t start = current_offset;
    size_t end = current_offset + size_bytes;
    current_offset = end;
    offsets.push_back(end);

    json += ",\"" + name + "\":{";
    json += "\"dtype\":\"F32\",";

    json += "\"shape\":[";
    for (size_t d = 0; d < t->shape().size(); ++d) {
      json += std::to_string(t->shape()[d]);
      if (d + 1 < t->shape().size())
        json += ",";
    }
    json += "],";

    json += "\"data_offsets\":[" + std::to_string(start) + "," +
            std::to_string(end) + "]";
    json += "}";
  }
  json += "}";
  return json;
}

static bool parse_json_offsets_shape_and_dtype(const std::string &json,
                                               const std::string &tensor_name,
                                               size_t &start, size_t &end,
                                               std::vector<size_t> &shape,
                                               std::string &dtype) {
  std::string search_key = "\"" + tensor_name + "\"";
  size_t name_pos = json.find(search_key);
  if (name_pos == std::string::npos)
    return false;

  size_t dtype_pos = json.find("\"dtype\"", name_pos);
  if (dtype_pos != std::string::npos) {
    size_t colon_pos = json.find(":", dtype_pos);
    if (colon_pos != std::string::npos) {
      size_t quote_start = json.find("\"", colon_pos);
      if (quote_start != std::string::npos) {
        size_t quote_end = json.find("\"", quote_start + 1);
        if (quote_end != std::string::npos) {
          dtype = json.substr(quote_start + 1, quote_end - quote_start - 1);
        }
      }
    }
  }

  size_t shape_pos = json.find("\"shape\"", name_pos);
  if (shape_pos != std::string::npos) {
    size_t start_bracket = json.find("[", shape_pos);
    size_t end_bracket = json.find("]", shape_pos);
    if (start_bracket != std::string::npos &&
        end_bracket != std::string::npos && start_bracket < end_bracket) {
      std::string shape_str =
          json.substr(start_bracket + 1, end_bracket - start_bracket - 1);
      shape.clear();
      size_t pos = 0;
      while (pos < shape_str.size()) {
        size_t next_comma = shape_str.find(",", pos);
        std::string dim_str = shape_str.substr(
            pos, (next_comma == std::string::npos) ? std::string::npos
                                                   : next_comma - pos);
        while (!dim_str.empty() && dim_str.front() == ' ')
          dim_str.erase(0, 1);
        while (!dim_str.empty() && dim_str.back() == ' ')
          dim_str.pop_back();
        if (!dim_str.empty()) {
          shape.push_back(std::stoull(dim_str));
        }
        if (next_comma == std::string::npos)
          break;
        pos = next_comma + 1;
      }
    }
  }

  size_t offset_pos = json.find("\"data_offsets\"", name_pos);
  if (offset_pos != std::string::npos) {
    size_t start_bracket = json.find("[", offset_pos);
    size_t end_bracket = json.find("]", offset_pos);
    if (start_bracket != std::string::npos &&
        end_bracket != std::string::npos && start_bracket < end_bracket) {
      std::string offset_str =
          json.substr(start_bracket + 1, end_bracket - start_bracket - 1);
      size_t comma = offset_str.find(",");
      if (comma != std::string::npos) {
        start = std::stoull(offset_str.substr(0, comma));
        end = std::stoull(offset_str.substr(comma + 1));
        return true;
      }
    }
  }
  return false;
}

static bool write_safetensors(
    const std::string &filepath,
    const std::vector<std::pair<std::string, Tensor *>> &tensors) {
  std::vector<size_t> offsets;
  std::string json = build_safetensors_json(tensors, offsets);

  size_t json_len = json.size();
  size_t padding = (8 - (json_len % 8)) % 8;
  if (padding > 0) {
    json.append(padding, ' ');
    json_len = json.size();
  }

  std::ofstream out(filepath, std::ios::binary);
  if (!out.is_open())
    return false;

  uint64_t header_size = json_len;
  out.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
  out.write(json.data(), json_len);

  for (size_t i = 0; i < tensors.size(); ++i) {
    const Tensor *t = tensors[i].second;
    out.write(reinterpret_cast<const char *>(t->data().data()),
              t->size() * sizeof(float));
  }

  return out.good();
}

static bool read_safetensors(const std::string &filepath,
                             const std::vector<std::pair<std::string, Tensor *>> &tensors) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in.is_open())
    return false;

  uint64_t header_size = 0;
  in.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  if (!in.good() || header_size == 0 || header_size > 1024 * 1024 * 10) {
    return false;
  }

  std::string json(header_size, '\0');
  in.read(&json[0], header_size);
  if (!in.good())
    return false;

  size_t payload_start = 8 + header_size;

  for (const auto &p : tensors) {
    const std::string &name = p.first;
    Tensor *t = p.second;

    size_t start = 0, end = 0;
    std::vector<size_t> shape;
    std::string dtype = "F32";
    if (!parse_json_offsets_shape_and_dtype(json, name, start, end, shape, dtype)) {
      std::cerr << "[WARNING] Checkpoint | Tensor '" << name
                << "' not found in header." << std::endl;
      continue;
    }

    if (shape != t->shape()) {
      std::cerr << "[WARNING] Checkpoint | Tensor '" << name
                << "' shape mismatch: expected ";
      for (auto s : t->shape())
        std::cerr << s << " ";
      std::cerr << ", got ";
      for (auto s : shape)
        std::cerr << s << " ";
      std::cerr << std::endl;
      continue;
    }

    in.seekg(payload_start + start);
    if (!in.good())
      return false;

    // Direct read assuming float32
    in.read(reinterpret_cast<char *>(t->data().data()),
            t->size() * sizeof(float));
    if (!in.good())
      return false;
  }

  return true;
}

// Reads the raw header size and JSON header contents from a safetensors file.
static bool read_safetensors_header(const std::string &filepath, std::string &json) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in.is_open()) return false;
  uint64_t header_size = 0;
  in.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  if (!in.good() || header_size == 0 || header_size > 1024 * 1024 * 10) {
    return false;
  }
  json.resize(header_size);
  in.read(&json[0], header_size);
  return in.good();
}

// Scans JSON header to check if it matches the Python format
static bool is_python_checkpoint(const std::string &json) {
  return json.find("\"tok_embeddings.weight\"") != std::string::npos;
}

// Loads a single Python tensor (with optional transposition and dtype adaptation)
static bool load_python_tensor(std::ifstream &in, size_t payload_start, const std::string &json,
                               const std::string &name, Tensor &dest, bool transpose = false) {
  size_t start = 0, end = 0;
  std::vector<size_t> shape;
  std::string dtype = "F32";
  if (!parse_json_offsets_shape_and_dtype(json, name, start, end, shape, dtype)) {
    std::cerr << "[WARNING] Checkpoint | Tensor '" << name << "' not found in header." << std::endl;
    return false;
  }
  
  size_t size_bytes = end - start;
  size_t num_elements = 0;
  std::vector<float> temp_buf;
  
  in.seekg(payload_start + start);
  if (!in.good()) return false;
  
  if (dtype == "BF16") {
    num_elements = size_bytes / 2;
    std::vector<uint16_t> raw_buf(num_elements);
    in.read(reinterpret_cast<char *>(raw_buf.data()), size_bytes);
    if (!in.good()) return false;
    temp_buf.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      temp_buf[i] = bf16_to_f32(raw_buf[i]);
    }
  } else if (dtype == "F16") {
    num_elements = size_bytes / 2;
    std::vector<uint16_t> raw_buf(num_elements);
    in.read(reinterpret_cast<char *>(raw_buf.data()), size_bytes);
    if (!in.good()) return false;
    temp_buf.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      temp_buf[i] = f16_to_f32(raw_buf[i]);
    }
  } else {
    num_elements = size_bytes / sizeof(float);
    temp_buf.resize(num_elements);
    in.read(reinterpret_cast<char *>(temp_buf.data()), size_bytes);
    if (!in.good()) return false;
  }
  
  if (transpose) {
    if (shape.size() != 2 || dest.shape().size() != 2 ||
        shape[0] != dest.shape()[1] || shape[1] != dest.shape()[0]) {
      std::cerr << "[ERROR] Checkpoint | Transpose shape mismatch for '" << name << "'" << std::endl;
      return false;
    }
    transpose_matrix(temp_buf.data(), dest.data().data(), shape[0], shape[1]);
  } else {
    if (temp_buf.size() != dest.size()) {
      std::cerr << "[ERROR] Checkpoint | Shape mismatch for '" << name << "': expected size " << dest.size() << ", got " << temp_buf.size() << std::endl;
      return false;
    }
    std::copy(temp_buf.begin(), temp_buf.end(), dest.data().begin());
  }
  return true;
}

// Loads a fused Python tensor, slices it row-by-row, and transposes/saves into destinations
static bool load_python_fused_tensor(std::ifstream &in, size_t payload_start, const std::string &json,
                                     const std::string &name,
                                     const std::vector<std::pair<Tensor *, size_t>> &slices) {
  size_t start = 0, end = 0;
  std::vector<size_t> shape;
  std::string dtype = "F32";
  if (!parse_json_offsets_shape_and_dtype(json, name, start, end, shape, dtype)) {
    std::cerr << "[WARNING] Checkpoint | Fused Tensor '" << name << "' not found in header." << std::endl;
    return false;
  }
  
  size_t size_bytes = end - start;
  size_t num_elements = 0;
  std::vector<float> temp_buf;
  
  in.seekg(payload_start + start);
  if (!in.good()) return false;
  
  if (dtype == "BF16") {
    num_elements = size_bytes / 2;
    std::vector<uint16_t> raw_buf(num_elements);
    in.read(reinterpret_cast<char *>(raw_buf.data()), size_bytes);
    if (!in.good()) return false;
    temp_buf.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      temp_buf[i] = bf16_to_f32(raw_buf[i]);
    }
  } else if (dtype == "F16") {
    num_elements = size_bytes / 2;
    std::vector<uint16_t> raw_buf(num_elements);
    in.read(reinterpret_cast<char *>(raw_buf.data()), size_bytes);
    if (!in.good()) return false;
    temp_buf.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      temp_buf[i] = f16_to_f32(raw_buf[i]);
    }
  } else {
    num_elements = size_bytes / sizeof(float);
    temp_buf.resize(num_elements);
    in.read(reinterpret_cast<char *>(temp_buf.data()), size_bytes);
    if (!in.good()) return false;
  }
  
  size_t cols = shape[1]; // e.g. hidden_dim (1024)
  
  for (const auto &slice_pair : slices) {
    Tensor &dest = *slice_pair.first;
    size_t start_row = slice_pair.second;
    size_t dest_cols = dest.shape()[1]; // output features / intermediate_dim
    size_t num_rows = dest_cols;
    
    std::vector<float> slice_buf(num_rows * cols);
    for (size_t r = 0; r < num_rows; ++r) {
      size_t global_row = start_row + r;
      std::copy(temp_buf.begin() + global_row * cols, temp_buf.begin() + (global_row + 1) * cols, slice_buf.begin() + r * cols);
    }
    
    transpose_matrix(slice_buf.data(), dest.data().data(), num_rows, cols);
  }
  return true;
}

// Parses Python/MLX model weight files
static bool load_python_model(const std::string &filepath, const std::string &json, Transformer &model) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in.is_open()) return false;
  
  uint64_t header_size = 0;
  in.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  size_t payload_start = 8 + header_size;
  
  if (!load_python_tensor(in, payload_start, json, "tok_embeddings.weight", model.token_embeddings(), false)) return false;
  if (!load_python_tensor(in, payload_start, json, "output.weight", model.output_projection(), true)) return false;
  if (!load_python_tensor(in, payload_start, json, "norm.weight", model.final_norm().weight(), false)) return false;
  
  for (size_t l = 0; l < model.layers().size(); ++l) {
    std::string prefix = "layers." + std::to_string(l) + ".";
    auto &layer = model.layers()[l];
    
    if (!load_python_tensor(in, payload_start, json, prefix + "attention_norm.weight", layer.attn_norm.weight(), false)) return false;
    if (!load_python_tensor(in, payload_start, json, prefix + "ffn_norm.weight", layer.ffn_norm.weight(), false)) return false;
    if (!load_python_tensor(in, payload_start, json, prefix + "attention.wo.weight", layer.attn.Wo(), true)) return false;
    if (!load_python_tensor(in, payload_start, json, prefix + "feed_forward.w3.weight", layer.w_down, true)) return false;
    
    // De-fuse wqkv
    std::vector<std::pair<Tensor *, size_t>> wqkv_slices = {
      {&layer.attn.Wq(), 0},
      {&layer.attn.Wk(), model.config().n_heads * model.config().head_dim},
      {&layer.attn.Wv(), (model.config().n_heads + model.config().n_kv_heads) * model.config().head_dim}
    };
    if (!load_python_fused_tensor(in, payload_start, json, prefix + "attention.wqkv.weight", wqkv_slices)) return false;
    
    // De-fuse w12
    std::vector<std::pair<Tensor *, size_t>> w12_slices = {
      {&layer.w_gate, 0},
      {&layer.w_up, layer.w_gate.shape()[1]}
    };
    if (!load_python_fused_tensor(in, payload_start, json, prefix + "feed_forward.w12.weight", w12_slices)) return false;
  }
  return true;
}

// Parses Python/MLX optimizer moment files
static bool load_python_optimizer(const std::string &opt_filepath, const std::string &json,
                                  Transformer &model, AdamWOptimizer &adamw) {
  std::ifstream in(opt_filepath, std::ios::binary);
  if (!in.is_open()) return false;
  
  uint64_t header_size = 0;
  in.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  size_t payload_start = 8 + header_size;
  
  // 1. Read step count
  size_t start = 0, end = 0;
  std::vector<size_t> shape;
  std::string dtype = "U64";
  if (parse_json_offsets_shape_and_dtype(json, "step", start, end, shape, dtype)) {
    uint64_t step_val = 0;
    in.seekg(payload_start + start);
    in.read(reinterpret_cast<char *>(&step_val), sizeof(uint64_t));
    if (in.good()) {
      adamw.step_count() = static_cast<size_t>(step_val);
    }
  } else {
    std::cerr << "[WARNING] Checkpoint | Optimizer 'step' not found in header. Restarting step count at 0." << std::endl;
    adamw.step_count() = 0;
  }
  
  auto &m_states = adamw.m_states();
  auto &v_states = adamw.v_states();
  
  // Token embeddings
  if (!load_python_tensor(in, payload_start, json, "tok_embeddings.weight.m", m_states[0], false)) return false;
  if (!load_python_tensor(in, payload_start, json, "tok_embeddings.weight.v", v_states[0], false)) return false;
  
  // Output projection
  if (!load_python_tensor(in, payload_start, json, "output.weight.m", m_states[1], true)) return false;
  if (!load_python_tensor(in, payload_start, json, "output.weight.v", v_states[1], true)) return false;
  
  for (size_t l = 0; l < model.layers().size(); ++l) {
    std::string prefix = "layers." + std::to_string(l) + ".";
    size_t layer_start_idx = 2 + l * 7;
    
    // w_down
    if (!load_python_tensor(in, payload_start, json, prefix + "feed_forward.w3.weight.m", m_states[layer_start_idx + 2], true)) return false;
    if (!load_python_tensor(in, payload_start, json, prefix + "feed_forward.w3.weight.v", v_states[layer_start_idx + 2], true)) return false;
    
    // attn.Wo
    if (!load_python_tensor(in, payload_start, json, prefix + "attention.wo.weight.m", m_states[layer_start_idx + 6], true)) return false;
    if (!load_python_tensor(in, payload_start, json, prefix + "attention.wo.weight.v", v_states[layer_start_idx + 6], true)) return false;
    
    // Fused wqkv
    std::vector<std::pair<Tensor *, size_t>> wqkv_m_slices = {
      {&m_states[layer_start_idx + 3], 0},
      {&m_states[layer_start_idx + 4], model.config().n_heads * model.config().head_dim},
      {&m_states[layer_start_idx + 5], (model.config().n_heads + model.config().n_kv_heads) * model.config().head_dim}
    };
    std::vector<std::pair<Tensor *, size_t>> wqkv_v_slices = {
      {&v_states[layer_start_idx + 3], 0},
      {&v_states[layer_start_idx + 4], model.config().n_heads * model.config().head_dim},
      {&v_states[layer_start_idx + 5], (model.config().n_heads + model.config().n_kv_heads) * model.config().head_dim}
    };
    if (!load_python_fused_tensor(in, payload_start, json, prefix + "attention.wqkv.weight.m", wqkv_m_slices)) return false;
    if (!load_python_fused_tensor(in, payload_start, json, prefix + "attention.wqkv.weight.v", wqkv_v_slices)) return false;
    
    // Fused w12
    std::vector<std::pair<Tensor *, size_t>> w12_m_slices = {
      {&m_states[layer_start_idx + 0], 0},
      {&m_states[layer_start_idx + 1], m_states[layer_start_idx + 0].shape()[1]}
    };
    std::vector<std::pair<Tensor *, size_t>> w12_v_slices = {
      {&v_states[layer_start_idx + 0], 0},
      {&v_states[layer_start_idx + 1], v_states[layer_start_idx + 0].shape()[1]}
    };
    if (!load_python_fused_tensor(in, payload_start, json, prefix + "feed_forward.w12.weight.m", w12_m_slices)) return false;
    if (!load_python_fused_tensor(in, payload_start, json, prefix + "feed_forward.w12.weight.v", w12_v_slices)) return false;
  }
  return true;
}

// Adapted build_safetensors_json that uses generic SavedTensor parameters
static std::string build_safetensors_json_adapted(
    const std::vector<SavedTensor> &tensors,
    std::vector<size_t> &offsets) {
  std::string json = "{";
  json += "\"__metadata__\":{\"format\":\"pt\"}";

  size_t current_offset = 0;
  offsets.clear();
  offsets.push_back(0);

  for (const auto &t : tensors) {
    size_t size_bytes = t.raw_data ? t.raw_size_bytes : (t.size * sizeof(float));
    size_t start = current_offset;
    size_t end = current_offset + size_bytes;
    current_offset = end;
    offsets.push_back(end);

    json += ",\"" + t.name + "\":{";
    json += "\"dtype\":\"" + t.dtype + "\",";

    json += "\"shape\":[";
    for (size_t d = 0; d < t.shape.size(); ++d) {
      json += std::to_string(t.shape[d]);
      if (d + 1 < t.shape.size())
        json += ",";
    }
    json += "],";

    json += "\"data_offsets\":[" + std::to_string(start) + "," +
            std::to_string(end) + "]";
    json += "}";
  }
  json += "}";
  return json;
}

// Adapted write_safetensors that supports raw data blocks
static bool write_safetensors_adapted(
    const std::string &filepath,
    const std::vector<SavedTensor> &tensors) {
  std::vector<size_t> offsets;
  std::string json = build_safetensors_json_adapted(tensors, offsets);

  size_t json_len = json.size();
  size_t padding = (8 - (json_len % 8)) % 8;
  if (padding > 0) {
    json.append(padding, ' ');
    json_len = json.size();
  }

  std::ofstream out(filepath, std::ios::binary);
  if (!out.is_open())
    return false;

  uint64_t header_size = json_len;
  out.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
  out.write(json.data(), json_len);

  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].raw_data) {
      out.write(tensors[i].raw_data, tensors[i].raw_size_bytes);
    } else {
      out.write(reinterpret_cast<const char *>(tensors[i].data),
                tensors[i].size * sizeof(float));
    }
  }

  return out.good();
}

// Save weights and optimizer state to files in Python/MLX-compatible format.
bool Checkpoint::save(const std::string &filepath, Transformer &model,
                      Optimizer &optimizer, size_t step) {
  std::vector<SavedTensor> save_list;
  
  // Embeddings
  {
    SavedTensor t;
    t.name = "tok_embeddings.weight";
    t.shape = model.token_embeddings().shape();
    t.data = model.token_embeddings().data().data();
    t.size = model.token_embeddings().size();
    save_list.push_back(t);
  }
  
  // Output projection (transposed)
  {
    SavedTensor t;
    t.name = "output.weight";
    t.shape = {model.output_projection().shape()[1], model.output_projection().shape()[0]};
    t.owned_data = transpose_to_vector(model.output_projection());
    t.data = t.owned_data.data();
    t.size = t.owned_data.size();
    save_list.push_back(t);
  }
  
  // Final norm
  {
    SavedTensor t;
    t.name = "norm.weight";
    t.shape = model.final_norm().weight().shape();
    t.data = model.final_norm().weight().data().data();
    t.size = model.final_norm().weight().size();
    save_list.push_back(t);
  }
  
  // Layers
  for (size_t l = 0; l < model.layers().size(); ++l) {
    std::string prefix = "layers." + std::to_string(l) + ".";
    auto &layer = model.layers()[l];
    
    // attn_norm
    {
      SavedTensor t;
      t.name = prefix + "attention_norm.weight";
      t.shape = layer.attn_norm.weight().shape();
      t.data = layer.attn_norm.weight().data().data();
      t.size = layer.attn_norm.weight().size();
      save_list.push_back(t);
    }
    
    // ffn_norm
    {
      SavedTensor t;
      t.name = prefix + "ffn_norm.weight";
      t.shape = layer.ffn_norm.weight().shape();
      t.data = layer.ffn_norm.weight().data().data();
      t.size = layer.ffn_norm.weight().size();
      save_list.push_back(t);
    }
    
    // attention.wo.weight (transposed)
    {
      SavedTensor t;
      t.name = prefix + "attention.wo.weight";
      t.shape = {layer.attn.Wo().shape()[1], layer.attn.Wo().shape()[0]};
      t.owned_data = transpose_to_vector(layer.attn.Wo());
      t.data = t.owned_data.data();
      t.size = t.owned_data.size();
      save_list.push_back(t);
    }
    
    // feed_forward.w3.weight (transposed)
    {
      SavedTensor t;
      t.name = prefix + "feed_forward.w3.weight";
      t.shape = {layer.w_down.shape()[1], layer.w_down.shape()[0]};
      t.owned_data = transpose_to_vector(layer.w_down);
      t.data = t.owned_data.data();
      t.size = t.owned_data.size();
      save_list.push_back(t);
    }
    
    // Fused attention.wqkv.weight (transposed and fused)
    {
      SavedTensor t;
      t.name = prefix + "attention.wqkv.weight";
      size_t q_features = layer.attn.Wq().shape()[1];
      size_t kv_features = layer.attn.Wk().shape()[1];
      size_t hidden_dim = layer.attn.Wq().shape()[0];
      t.shape = {q_features + 2 * kv_features, hidden_dim};
      t.owned_data = fuse_and_transpose_wqkv(layer.attn.Wq(), layer.attn.Wk(), layer.attn.Wv());
      t.data = t.owned_data.data();
      t.size = t.owned_data.size();
      save_list.push_back(t);
    }
    
    // Fused feed_forward.w12.weight (transposed and fused)
    {
      SavedTensor t;
      t.name = prefix + "feed_forward.w12.weight";
      size_t intermediate_dim = layer.w_gate.shape()[1];
      size_t hidden_dim = layer.w_gate.shape()[0];
      t.shape = {2 * intermediate_dim, hidden_dim};
      t.owned_data = fuse_and_transpose_w12(layer.w_gate, layer.w_up);
      t.data = t.owned_data.data();
      t.size = t.owned_data.size();
      save_list.push_back(t);
    }
  }
  
  if (!write_safetensors_adapted(filepath, save_list)) {
    return false;
  }
  
  // 2. Save optimizer state if using AdamW
  AdamWOptimizer *adamw = dynamic_cast<AdamWOptimizer *>(&optimizer);
  if (adamw) {
    std::string opt_filepath = filepath;
    size_t pos = opt_filepath.find(".safetensors");
    if (pos != std::string::npos) {
      opt_filepath.replace(pos, 12, ".opt.safetensors");
    } else {
      opt_filepath += ".opt.safetensors";
    }
    
    std::vector<SavedTensor> opt_save_list;
    
    // Step count
    uint64_t step_count_val = adamw->step_count();
    SavedTensor t_step;
    t_step.name = "step";
    t_step.shape = {};
    t_step.dtype = "U64";
    t_step.raw_data = reinterpret_cast<const char *>(&step_count_val);
    t_step.raw_size_bytes = sizeof(uint64_t);
    opt_save_list.push_back(t_step);
    
    auto &m_states = adamw->m_states();
    auto &v_states = adamw->v_states();
    
    // Token embeddings
    {
      SavedTensor t_m, t_v;
      t_m.name = "tok_embeddings.weight.m";
      t_m.shape = m_states[0].shape();
      t_m.data = m_states[0].data().data();
      t_m.size = m_states[0].size();
      
      t_v.name = "tok_embeddings.weight.v";
      t_v.shape = v_states[0].shape();
      t_v.data = v_states[0].data().data();
      t_v.size = v_states[0].size();
      
      opt_save_list.push_back(t_m);
      opt_save_list.push_back(t_v);
    }
    
    // Output projection (transposed)
    {
      SavedTensor t_m, t_v;
      t_m.name = "output.weight.m";
      t_m.shape = {m_states[1].shape()[1], m_states[1].shape()[0]};
      t_m.owned_data = transpose_to_vector(m_states[1]);
      t_m.data = t_m.owned_data.data();
      t_m.size = t_m.owned_data.size();
      
      t_v.name = "output.weight.v";
      t_v.shape = {v_states[1].shape()[1], v_states[1].shape()[0]};
      t_v.owned_data = transpose_to_vector(v_states[1]);
      t_v.data = t_v.owned_data.data();
      t_v.size = t_v.owned_data.size();
      
      opt_save_list.push_back(t_m);
      opt_save_list.push_back(t_v);
    }
    
    // Fictional / dummy norms moments (to keep Python tree loading flawless)
    std::vector<float> norm_zeros(model.config().hidden_dim, 0.0f);
    {
      SavedTensor t_m, t_v;
      t_m.name = "norm.weight.m";
      t_m.shape = {model.config().hidden_dim};
      t_m.data = norm_zeros.data();
      t_m.size = norm_zeros.size();
      
      t_v.name = "norm.weight.v";
      t_v.shape = {model.config().hidden_dim};
      t_v.data = norm_zeros.data();
      t_v.size = norm_zeros.size();
      
      opt_save_list.push_back(t_m);
      opt_save_list.push_back(t_v);
    }
    
    for (size_t l = 0; l < model.layers().size(); ++l) {
      std::string prefix = "layers." + std::to_string(l) + ".";
      size_t layer_start_idx = 2 + l * 7;
      auto &layer = model.layers()[l];
      
      // ffn dummy norms
      {
        SavedTensor t_m, t_v;
        t_m.name = prefix + "ffn_norm.weight.m";
        t_m.shape = {model.config().hidden_dim};
        t_m.data = norm_zeros.data();
        t_m.size = norm_zeros.size();
        
        t_v.name = prefix + "ffn_norm.weight.v";
        t_v.shape = {model.config().hidden_dim};
        t_v.data = norm_zeros.data();
        t_v.size = norm_zeros.size();
        
        opt_save_list.push_back(t_m);
        opt_save_list.push_back(t_v);
      }
      
      // attn dummy norms
      {
        SavedTensor t_m, t_v;
        t_m.name = prefix + "attention_norm.weight.m";
        t_m.shape = {model.config().hidden_dim};
        t_m.data = norm_zeros.data();
        t_m.size = norm_zeros.size();
        
        t_v.name = prefix + "attention_norm.weight.v";
        t_v.shape = {model.config().hidden_dim};
        t_v.data = norm_zeros.data();
        t_v.size = norm_zeros.size();
        
        opt_save_list.push_back(t_m);
        opt_save_list.push_back(t_v);
      }
      
      // w_down (transposed)
      {
        SavedTensor t_m, t_v;
        t_m.name = prefix + "feed_forward.w3.weight.m";
        t_m.shape = {m_states[layer_start_idx + 2].shape()[1], m_states[layer_start_idx + 2].shape()[0]};
        t_m.owned_data = transpose_to_vector(m_states[layer_start_idx + 2]);
        t_m.data = t_m.owned_data.data();
        t_m.size = t_m.owned_data.size();
        
        t_v.name = prefix + "feed_forward.w3.weight.v";
        t_v.shape = {v_states[layer_start_idx + 2].shape()[1], v_states[layer_start_idx + 2].shape()[0]};
        t_v.owned_data = transpose_to_vector(v_states[layer_start_idx + 2]);
        t_v.data = t_v.owned_data.data();
        t_v.size = t_v.owned_data.size();
        
        opt_save_list.push_back(t_m);
        opt_save_list.push_back(t_v);
      }
      
      // attn.Wo (transposed)
      {
        SavedTensor t_m, t_v;
        t_m.name = prefix + "attention.wo.weight.m";
        t_m.shape = {m_states[layer_start_idx + 6].shape()[1], m_states[layer_start_idx + 6].shape()[0]};
        t_m.owned_data = transpose_to_vector(m_states[layer_start_idx + 6]);
        t_m.data = t_m.owned_data.data();
        t_m.size = t_m.owned_data.size();
        
        t_v.name = prefix + "attention.wo.weight.v";
        t_v.shape = {v_states[layer_start_idx + 6].shape()[1], v_states[layer_start_idx + 6].shape()[0]};
        t_v.owned_data = transpose_to_vector(v_states[layer_start_idx + 6]);
        t_v.data = t_v.owned_data.data();
        t_v.size = t_v.owned_data.size();
        
        opt_save_list.push_back(t_m);
        opt_save_list.push_back(t_v);
      }
      
      // Fused wqkv
      {
        SavedTensor t_m, t_v;
        t_m.name = prefix + "attention.wqkv.weight.m";
        size_t q_features = layer.attn.Wq().shape()[1];
        size_t kv_features = layer.attn.Wk().shape()[1];
        size_t hidden_dim = layer.attn.Wq().shape()[0];
        t_m.shape = {q_features + 2 * kv_features, hidden_dim};
        t_m.owned_data = fuse_and_transpose_wqkv(m_states[layer_start_idx + 3], m_states[layer_start_idx + 4], m_states[layer_start_idx + 5]);
        t_m.data = t_m.owned_data.data();
        t_m.size = t_m.owned_data.size();
        
        t_v.name = prefix + "attention.wqkv.weight.v";
        t_v.shape = {q_features + 2 * kv_features, hidden_dim};
        t_v.owned_data = fuse_and_transpose_wqkv(v_states[layer_start_idx + 3], v_states[layer_start_idx + 4], v_states[layer_start_idx + 5]);
        t_v.data = t_v.owned_data.data();
        t_v.size = t_v.owned_data.size();
        
        opt_save_list.push_back(t_m);
        opt_save_list.push_back(t_v);
      }
      
      // Fused w12
      {
        SavedTensor t_m, t_v;
        t_m.name = prefix + "feed_forward.w12.weight.m";
        size_t intermediate_dim = layer.w_gate.shape()[1];
        size_t hidden_dim = layer.w_gate.shape()[0];
        t_m.shape = {2 * intermediate_dim, hidden_dim};
        t_m.owned_data = fuse_and_transpose_w12(m_states[layer_start_idx + 0], m_states[layer_start_idx + 1]);
        t_m.data = t_m.owned_data.data();
        t_m.size = t_m.owned_data.size();
        
        t_v.name = prefix + "feed_forward.w12.weight.v";
        t_v.shape = {2 * intermediate_dim, hidden_dim};
        t_v.owned_data = fuse_and_transpose_w12(v_states[layer_start_idx + 0], v_states[layer_start_idx + 1]);
        t_v.data = t_v.owned_data.data();
        t_v.size = t_v.owned_data.size();
        
        opt_save_list.push_back(t_m);
        opt_save_list.push_back(t_v);
      }
    }
    
    if (!write_safetensors_adapted(opt_filepath, opt_save_list)) {
      return false;
    }
  }
  
  return true;
}

// Load weights and optimizer state from files. Automatically detects Python vs native C++ layouts and formats.
bool Checkpoint::load(const std::string &filepath, Transformer &model,
                      Optimizer &optimizer, size_t &step) {
  std::string json;
  if (!read_safetensors_header(filepath, json)) {
    return false;
  }
  
  if (is_python_checkpoint(json)) {
    std::cout << "[INFO] Checkpoint | Detected Python/MLX formatted checkpoint. Using compatibility adapter." << std::endl;
    if (!load_python_model(filepath, json, model)) {
      return false;
    }
  } else {
    std::cout << "[INFO] Checkpoint | Detected Native C++ formatted checkpoint." << std::endl;
    auto model_tensors = get_model_tensors(model);
    if (!read_safetensors(filepath, model_tensors)) {
      return false;
    }
  }

  // 2. Load optimizer state if using AdamW
  AdamWOptimizer *adamw = dynamic_cast<AdamWOptimizer *>(&optimizer);
  if (adamw) {
    std::string opt_filepath = filepath;
    size_t pos = opt_filepath.find(".safetensors");
    if (pos != std::string::npos) {
      opt_filepath.replace(pos, 12, ".opt.safetensors");
    } else {
      opt_filepath += ".opt.safetensors";
    }

    std::string opt_json;
    if (read_safetensors_header(opt_filepath, opt_json)) {
      if (is_python_checkpoint(opt_json) || opt_json.find("\"step\"") != std::string::npos) {
        if (!load_python_optimizer(opt_filepath, opt_json, model, *adamw)) {
          std::cerr << "[WARNING] Checkpoint | Failed to parse Python optimizer state. Adam moments will restart from zero." << std::endl;
          step = 0;
        } else {
          step = adamw->step_count();
          std::cout << "[INFO] Resumed optimizer step count: " << step << std::endl;
        }
      } else {
        auto param_names = get_optimizer_param_names(model);
        auto &m_states = adamw->m_states();
        auto &v_states = adamw->v_states();

        std::vector<std::pair<std::string, Tensor *>> opt_tensors;
        Tensor step_tensor({1}, 0.0f);
        opt_tensors.push_back({"step", &step_tensor});

        for (size_t i = 0; i < param_names.size(); ++i) {
          opt_tensors.push_back({"m." + param_names[i], &m_states[i]});
          opt_tensors.push_back({"v." + param_names[i], &v_states[i]});
        }

        if (read_safetensors(opt_filepath, opt_tensors)) {
          adamw->step_count() = static_cast<size_t>(step_tensor(0));
          step = adamw->step_count();
          std::cout << "[INFO] Resumed optimizer step count: " << step << std::endl;
        } else {
          std::cerr << "[WARNING] Checkpoint | Failed to read native optimizer state. Adam moments will restart from zero." << std::endl;
          step = 0;
        }
      }
    } else {
      std::cerr << "[WARNING] Checkpoint | Failed to read optimizer state header. Adam moments will restart from zero." << std::endl;
      step = 0;
    }
  }

  return true;
}
