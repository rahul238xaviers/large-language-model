#!/usr/bin/env python3
"""Convert a Python/MLX safetensors checkpoint to native C++ format.

Usage:
    python convert_checkpoint.py <python_run_dir> <output_dir>

Reads step_NNNNNN.safetensors (+ .opt.safetensors) from the Python run directory
and writes native C++ checkpoint files to output_dir/checkpoints/.
"""

import json
import os
import re
import struct
import sys
import numpy as np


def bf16_to_f32(bf16_val: int) -> float:
    """Convert bf16 uint16 to float32."""
    bits = np.uint32(bf16_val) << np.uint32(16)
    return float(np.frombuffer(bits.tobytes(), dtype=np.float32)[0])


def read_safetensors(filepath: str) -> dict:
    """Read a safetensors file and return {name: numpy_array}."""
    with open(filepath, "rb") as f:
        header_size = struct.unpack("Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            dtype_str = info["dtype"]
            shape = info["shape"]
            f.seek(8 + header_size + start)
            raw = f.read(end - start)
            if dtype_str == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).astype(np.float32)
                # bf16 → f32: left-shift by 16 bits
                arr = arr.view(np.uint32) << 16
                arr = arr.view(np.float32)
            elif dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            elif dtype_str == "U64":
                arr = np.frombuffer(raw, dtype=np.uint64)
            elif dtype_str == "F64":
                arr = np.frombuffer(raw, dtype=np.float64)
            else:
                raise ValueError(f"Unknown dtype: {dtype_str}")
            arr = arr.reshape(shape)
            tensors[name] = arr
    return tensors


def write_safetensors_native(filepath: str, tensors: dict):
    """Write tensors in native C++ format (dtype=F32, row-major)."""
    offsets = {}
    current_offset = 0
    header_dict = {}
    for name in sorted(tensors.keys()):
        arr = tensors[name]
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        nbytes = arr.nbytes
        header_dict[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + nbytes],
        }
        offsets[name] = (current_offset, current_offset + nbytes)
        current_offset += nbytes

    header_json = json.dumps(header_dict, separators=(",", ":"))
    json_bytes = header_json.encode("utf-8")
    padding = (8 - (len(json_bytes) % 8)) % 8
    json_bytes += b" " * padding
    header_size = len(json_bytes)

    with open(filepath, "wb") as f:
        f.write(struct.pack("Q", header_size))
        f.write(json_bytes)
        for name in sorted(tensors.keys()):
            arr = tensors[name]
            f.write(arr.tobytes())


def convert_model_checkpoint(src_ckpt: str, dst_ckpt: str, hidden_dim=1024,
                              intermediate_dim=2730, n_heads=16, n_kv_heads=8):
    """Convert MLX checkpoint to native C++ format.

    The native C++ format uses the following tensor names (from Checkpoint.cpp):
      token_embeddings, output_projection, final_norm.weight
      layers.N.attn_norm.weight, layers.N.attn.Wq, layers.N.attn.Wk,
      layers.N.attn.Wv, layers.N.attn.Wo, layers.N.ffn_norm.weight,
      layers.N.w_gate, layers.N.w_up, layers.N.w_down
    """
    print(f"Reading Python checkpoint from {src_ckpt}")
    ckpt = read_safetensors(src_ckpt)

    result = {}
    q_dim = n_heads * (hidden_dim // n_heads)
    kv_dim = n_kv_heads * (hidden_dim // n_heads)

    # token_embeddings: [V, H] → stored as-is in C++
    result["token_embeddings"] = ckpt["tok_embeddings.weight"].astype(np.float32)

    # output_projection: Python [V, H], C++ [H, V] (transposed)
    result["output_projection"] = ckpt["output.weight"].astype(np.float32).T

    # final_norm.weight
    result["final_norm.weight"] = ckpt["norm.weight"].astype(np.float32)

    for layer_idx in range(24):
        prefix = f"layers.{layer_idx}."
        print(f"  Converting layer {layer_idx}...")

        # attn_norm.weight
        result[f"{prefix}attn_norm.weight"] = ckpt[f"{prefix}attention_norm.weight"].astype(np.float32)
        # ffn_norm.weight
        result[f"{prefix}ffn_norm.weight"] = ckpt[f"{prefix}ffn_norm.weight"].astype(np.float32)

        # attn.Wo: Python [H, H], C++ [H, H] (transposed from checkpoint)
        wo = ckpt[f"{prefix}attention.wo.weight"].astype(np.float32)
        result[f"{prefix}attn.Wo"] = wo.T

        # w_down: Python [H, I] = w3 stored as [2730, 1024] in C++ (transposed)
        w3 = ckpt[f"{prefix}feed_forward.w3.weight"].astype(np.float32)  # [H, I]
        result[f"{prefix}w_down"] = w3.T  # [I, H]

        # De-fuse attention.wqkv → attn.Wq, attn.Wk, attn.Wv
        wqkv = ckpt[f"{prefix}attention.wqkv.weight"].astype(np.float32)  # [2048, 1024]
        wq = wqkv[:q_dim, :].T       # [H, H]
        wk = wqkv[q_dim:q_dim + kv_dim, :].T   # [H, 512]
        wv = wqkv[q_dim + kv_dim:, :].T  # [H, 512]
        result[f"{prefix}attn.Wq"] = wq
        result[f"{prefix}attn.Wk"] = wk
        result[f"{prefix}attn.Wv"] = wv

        # De-fuse feed_forward.w12 → w_gate, w_up
        w12 = ckpt[f"{prefix}feed_forward.w12.weight"].astype(np.float32)  # [5460, 1024]
        w_gate = w12[:intermediate_dim, :].T  # [H, I]
        w_up = w12[intermediate_dim:, :].T    # [H, I]
        result[f"{prefix}w_gate"] = w_gate
        result[f"{prefix}w_up"] = w_up

    os.makedirs(os.path.dirname(dst_ckpt), exist_ok=True)
    write_safetensors_native(dst_ckpt, result)
    print(f"Wrote native checkpoint to {dst_ckpt}")


def convert_optimizer_checkpoint(src_opt: str, dst_opt: str, hidden_dim=1024,
                                  intermediate_dim=2730, n_heads=16, n_kv_heads=8):
    """Convert MLX optimizer state to native C++ format."""
    print(f"Reading Python optimizer from {src_opt}")
    opt = read_safetensors(src_opt)

    result = {}

    # Step count
    step_val = opt.get("step", np.array([30000], dtype=np.uint64))
    if step_val.ndim == 0 or step_val.size == 1:
        step_int = int(step_val.item() if hasattr(step_val, 'item') else step_val[0])
    else:
        step_int = int(step_val[0])
    step_tensor = np.array([float(step_int)], dtype=np.float32)
    result["step"] = step_tensor

    # Build parameter names matching the C++ optimizer registration order
    # From get_optimizer_param_names in Checkpoint.cpp:
    #   token_embeddings, output_projection
    #   for each layer: w_gate, w_up, w_down, attn.Wq, attn.Wk, attn.Wv, attn.Wo
    param_names = []
    param_names.append("token_embeddings")
    param_names.append("output_projection")
    for l in range(24):
        prefix = f"layers.{l}."
        param_names.append(prefix + "w_gate")
        param_names.append(prefix + "w_up")
        param_names.append(prefix + "w_down")
        param_names.append(prefix + "attn.Wq")
        param_names.append(prefix + "attn.Wk")
        param_names.append(prefix + "attn.Wv")
        param_names.append(prefix + "attn.Wo")

    q_dim = n_heads * (hidden_dim // n_heads)
    kv_dim = n_kv_heads * (hidden_dim // n_heads)

    # Map C++ parameter base names back to the fused Python optimizer tensors.
    fused_map = {
        "Wq": "attention.wqkv.weight",
        "Wk": "attention.wqkv.weight",
        "Wv": "attention.wqkv.weight",
        "Wo": "attention.wo.weight",
        "w_gate": "feed_forward.w12.weight",
        "w_up": "feed_forward.w12.weight",
        "w_down": "feed_forward.w3.weight",
    }

    def slice_fused(fused_arr, base):
        """Slice/de-transpose a fused Python moment tensor into a native C++ param."""
        if base in ("Wq", "Wk", "Wv"):
            wq = fused_arr[:q_dim, :].T
            wk = fused_arr[q_dim:q_dim + kv_dim, :].T
            wv = fused_arr[q_dim + kv_dim:, :].T
            return wq if base == "Wq" else (wk if base == "Wk" else wv)
        if base in ("w_gate", "w_up"):
            w_g = fused_arr[:intermediate_dim, :].T
            w_u = fused_arr[intermediate_dim:, :].T
            return w_g if base == "w_gate" else w_u
        # "Wo" / "w_down": plain transposed weight moments
        return fused_arr.T

    for pname in param_names:
        base = pname.rsplit(".", 1)[-1] if "." in pname else pname

        # Unfused embeddings / output projection moments
        if base == "token_embeddings":
            result["m.token_embeddings"] = opt["tok_embeddings.weight.m"].astype(np.float32)
            result["v.token_embeddings"] = opt["tok_embeddings.weight.v"].astype(np.float32)
            continue
        if base == "output_projection":
            result["m.output_projection"] = opt["output.weight.m"].astype(np.float32).T
            result["v.output_projection"] = opt["output.weight.v"].astype(np.float32).T
            continue

        fused_name = fused_map.get(base)
        if not fused_name:
            continue
        # Python fused tensors live under the layer prefix (e.g. "layers.0."),
        # regardless of the native param being "layers.0.attn.Wq" or "layers.0.w_gate".
        m = re.match(r"layers\.(\d+)\.", pname)
        if not m:
            continue
        layer_prefix = "layers.%s." % m.group(1)
        fused_key = layer_prefix + fused_name
        m_src = fused_key + ".m"
        v_src = fused_key + ".v"
        if m_src in opt:
            result["m." + pname] = slice_fused(opt[m_src].astype(np.float32), base)
        if v_src in opt:
            result["v." + pname] = slice_fused(opt[v_src].astype(np.float32), base)

    os.makedirs(os.path.dirname(dst_opt), exist_ok=True)
    write_safetensors_native(dst_opt, result)
    print(f"Wrote native optimizer to {dst_opt}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <python_run_dir> <output_dir>")
        sys.exit(1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    # Find the latest checkpoint
    ckpt_dir = os.path.join(src_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = src_dir

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors") and ".opt." not in f]
    if not ckpt_files:
        print(f"No checkpoint files found in {ckpt_dir}")
        sys.exit(1)

    # Use the highest step number
    def step_num(f):
        try:
            return int(f.replace("step_", "").replace(".safetensors", ""))
        except:
            return 0

    latest = max(ckpt_files, key=step_num)
    step = step_num(latest)
    src_ckpt = os.path.join(ckpt_dir, latest)
    src_opt = os.path.join(ckpt_dir, f"step_{step:06d}.opt.safetensors")

    dst_ckpt_dir = os.path.join(dst_dir, "checkpoints")
    os.makedirs(dst_ckpt_dir, exist_ok=True)
    dst_ckpt = os.path.join(dst_ckpt_dir, f"step_{step:07d}.safetensors")
    dst_opt = os.path.join(dst_ckpt_dir, f"step_{step:07d}.opt.safetensors")

    convert_model_checkpoint(src_ckpt, dst_ckpt)
    if os.path.exists(src_opt):
        convert_optimizer_checkpoint(src_opt, dst_opt)
    else:
        print(f"[WARNING] Optimizer state not found at {src_opt}, skipping")

    print(f"\nDone! Native checkpoint saved to {dst_dir}/")
    print(f"To continue training from step {step} with this converted model:")
    print(f"  ./build/run_trainer --batch_size 32 --max_steps {step + 500} \\")
    print(f"    --checkpoint_interval 500 --init_checkpoint {dst_ckpt}")
