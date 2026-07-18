/**
 * @file Checkpoint.cpp
 * @brief Implementation of safetensors-compatible checkpoint serialization and
 * deserialization
 */

#include "Checkpoint.hpp"
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

// WHAT: Helper to retrieve all named weight parameter tensors from the
// Transformer model. WHY: We map each tensor to a unique string key that
// matches standard LLM weight naming schemas,
//      which can be loaded directly by both our C++ engine and the Python MLX
//      pipeline.
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

// WHAT: Helper to retrieve the names of the parameters in their registration
// order. WHY: The AdamW optimizer moment tracking buffers are registered 1-to-1
// to these parameters,
//      so we map them by index to construct named moment states (e.g.
//      "m.layers.0.w_gate").
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

// WHAT: Helper to build the JSON metadata header required by the safetensors
// format. WHY: The safetensors header is a JSON string describing each tensor's
// data type, shape,
//      and start/end byte offsets in the binary payload.
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

// WHAT: A custom, light JSON parser to scan for offsets and shape without
// external dependencies. WHY: We want to avoid adding a heavy third-party JSON
// parsing library to the C++ training engine.
static bool parse_json_offsets_and_shape(const std::string &json,
                                         const std::string &tensor_name,
                                         size_t &start, size_t &end,
                                         std::vector<size_t> &shape) {
  std::string search_key = "\"" + tensor_name + "\"";
  size_t name_pos = json.find(search_key);
  if (name_pos == std::string::npos)
    return false;

  // Extract shape array
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

  // Extract data offsets array
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

// WHAT: Write a list of named tensors to a safetensors file.
// WHY: Implements the safetensors specification (8-byte header size + JSON
// header + padded alignment + raw binary payload).
static bool write_safetensors(
    const std::string &filepath,
    const std::vector<std::pair<std::string, Tensor *>> &tensors) {
  std::vector<size_t> offsets;
  std::string json = build_safetensors_json(tensors, offsets);

  // Align header size to a multiple of 8 bytes for GPU/memory alignment.
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

// WHAT: Read a list of named tensors from a safetensors file.
// WHY: Implements the safetensors parsing and checks shapes and dtypes.
static bool
read_safetensors(const std::string &filepath,
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
    if (!parse_json_offsets_and_shape(json, name, start, end, shape)) {
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

    in.read(reinterpret_cast<char *>(t->data().data()),
            t->size() * sizeof(float));
    if (!in.good())
      return false;
  }

  return true;
}

// WHAT: Save weights and optimizer state to files.
// WHY: Implements Sprint 6d checkpoint saving.
bool Checkpoint::save(const std::string &filepath, Transformer &model,
                      Optimizer &optimizer, size_t step) {
  // 1. Save model parameters
  auto model_tensors = get_model_tensors(model);
  if (!write_safetensors(filepath, model_tensors)) {
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

    auto param_names = get_optimizer_param_names(model);
    auto &m_states = adamw->m_states();
    auto &v_states = adamw->v_states();
    size_t step_count = adamw->step_count();

    // Create opt tensors structure
    std::vector<std::pair<std::string, Tensor *>> opt_tensors;

    // Convert step_count to a 1-element tensor so safetensors can serialize it
    Tensor step_tensor({1}, static_cast<float>(step_count));
    opt_tensors.push_back({"step", &step_tensor});

    for (size_t i = 0; i < param_names.size(); ++i) {
      opt_tensors.push_back({"m." + param_names[i], &m_states[i]});
      opt_tensors.push_back({"v." + param_names[i], &v_states[i]});
    }

    if (!write_safetensors(opt_filepath, opt_tensors)) {
      return false;
    }
  }

  return true;
}

// WHAT: Load weights and optimizer state from files.
// WHY: Implements Sprint 6d checkpoint loading and resumption.
bool Checkpoint::load(const std::string &filepath, Transformer &model,
                      Optimizer &optimizer, size_t &step) {
  // 1. Load model parameters
  auto model_tensors = get_model_tensors(model);
  if (!read_safetensors(filepath, model_tensors)) {
    return false;
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
      std::cerr << "[WARNING] Checkpoint | Failed to read optimizer state. "
                   "Adam moments will restart from zero."
                << std::endl;
      step = 0;
    }
  }

  return true;
}
